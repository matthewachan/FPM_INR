# Main script for FPM-INR reconstruction
# Written by Haowen Zhou and Brandon Y. Feng
# Last modified on 10/26/2023
# Contact: Haowen Zhou (hzhou7@caltech.edu)


import os
import tqdm
import mat73
import scipy.io as sio
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn.functional as F

from network import FullModel
from utils import save_model_with_required_grad

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sub_spectrum(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, mag):
    O = torch.fft.fftshift(torch.fft.fft2(img_complex))
    to_pad_x = (spectrum_mask.shape[-2] * mag - O.shape[-2]) // 2
    to_pad_y = (spectrum_mask.shape[-1] * mag - O.shape[-1]) // 2
    O = F.pad(O, (to_pad_x, to_pad_x, to_pad_y, to_pad_y, 0, 0), "constant", 0)

    O_sub = torch.stack(
        [O[:, x_0[i] : x_1[i], y_0[i] : y_1[i]] for i in range(len(led_num))], dim=1
    )
    O_sub = O_sub * spectrum_mask
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    oI_sub = torch.abs(o_sub)

    return oI_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=15, type=int)
    parser.add_argument("--lr_decay_step", default=6, type=int)
    parser.add_argument("--num_feats", default=32, type=int)
    parser.add_argument("--num_modes", default=512, type=int)
    parser.add_argument("--c2f", default=False, action="store_true")
    parser.add_argument("--layer_norm", default=False, action="store_true")
    parser.add_argument("--amp", default=True, action="store_true")
    parser.add_argument("--sample", default="bloodsmear", type=str)
    parser.add_argument("--color", default="r", type=str)
    parser.add_argument("--is_system", default="Linux", type=str)  # "Windows". "Linux"

    args = parser.parse_args()

    num_epochs = args.num_epochs
    num_feats = args.num_feats
    num_modes = args.num_modes
    lr_decay_step = args.lr_decay_step
    use_c2f = args.c2f
    use_layernorm = args.layer_norm
    use_amp = args.amp

    sample = args.sample
    color = args.color
    is_os = args.is_system

    vis_dir = f"./vis/feat{num_feats}"

    os.makedirs(vis_dir, exist_ok=True)

    # Load data

    data_struct = sio.loadmat(
        "/nfshomes/mattchan/scratch/Fourier-Ptychography/Data/bloodsmear_blue.mat"
    )
    # MAGimg = 3
    # if sample == "Siemens":
    #     data_struct = sio.loadmat(f"data/{sample}/{sample}_{color}.mat")
    #     MAGimg = 3

    # else:
    #     data_struct = mat73.loadmat(f"data/{sample}/{sample}_{color}.mat")
    #     MAGimg = 2

    I = data_struct["imlow_HDR"].astype("float32")

    # Select ROI
    I = I[0 : int(num_modes), 0 : int(num_modes), :]  #######################

    # Raw measurement sidelength
    M = I.shape[0]
    N = I.shape[1]
    ID_len = I.shape[2]

    # NAx NAy
    # NAs = data_struct["na_calib"].astype("float32")
    # NAx = NAs[:, 0]
    # NAy = NAs[:, 1]
    illum = sio.loadmat("/nfshomes/mattchan/scratch/Fourier-Ptychography/na_illum.mat")
    NAx = illum["kx"].astype("float32")[0]
    NAy = illum["ky"].astype("float32")[0]

    # LED central wavelength
    if color == "r":
        wavelength = 0.632  # um
    elif color == "g":
        wavelength = 0.5126  # um
    elif color == "b":
        wavelength = 0.475  # um

    # Distance between two adjacent LEDs (unit: um)
    D_led = 4000
    # free-space k-vector
    k0 = 2 * np.pi / wavelength
    # Objective lens magnification
    mag = 2.0
    # Camera pixel pitch (unit: um)
    pixel_size = 1.845
    # pixel size at image plane (unit: um)
    D_pixel = pixel_size / mag
    # Objective lens NA
    NA = 0.1
    # Maximum k-value
    kmax = NA * k0

    # Calculate upsampliing ratio
    MAGimg = 4
    # Upsampled pixel count
    MM = int(M * MAGimg)
    NN = int(N * MAGimg)

    # Define spatial frequency coordinates
    Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
    Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
    Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)

    # Calculate illumination NA
    u = -NAx
    v = -NAy
    NAillu = np.sqrt(u**2 + v**2)
    order = np.argsort(NAillu)
    u = u[order]
    v = v[order]

    # NA shift in pixel from different LED illuminations
    ledpos_true = np.zeros((ID_len, 2), dtype=int)
    count = 0
    for idx in range(ID_len):
        Fx1_temp = np.abs(Fxx1 - k0 * u[idx])
        ledpos_true[count, 0] = np.argmin(Fx1_temp)
        Fy1_temp = np.abs(Fyy1 - k0 * v[idx])
        ledpos_true[count, 1] = np.argmin(Fy1_temp)
        count += 1
    # Raw measurements
    Isum = I[:, :, order] / np.max(I)

    # Define angular spectrum
    if sample == "Siemens":
        kxx, kyy = np.meshgrid(Fxx1[0, :M], Fxx1[0, :N])
    else:
        kxx, kyy = np.meshgrid(Fxx1[:M], Fxx1[:N])
    kxx, kyy = kxx - np.mean(kxx), kyy - np.mean(kyy)
    krr = np.sqrt(kxx**2 + kyy**2)
    mask_k = k0**2 - krr**2 > 0
    kzz_ampli = mask_k * np.abs(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz_phase = np.angle(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz = kzz_ampli * np.exp(1j * kzz_phase)

    # Define Pupil support
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax**2)] = 1

    Pupil0 = (
        torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1]).to(device)
    )
    kzz = torch.from_numpy(kzz).to(device).unsqueeze(0)
    Isum = torch.from_numpy(Isum).to(device)

    z_min = 0.0
    z_max = 1.0

    # Define LED Batch size
    led_batch_size = 1
    cur_ds = 1
    if use_c2f:
        c2f_sche = (
            [4] * (num_epochs // 5) + [2] * (num_epochs // 5) + [1] * (num_epochs // 5)
        )
        cur_ds = c2f_sche[0]

    model = FullModel(
        w=MM,
        h=MM,
        num_feats=num_feats,
        x_mode=num_modes,
        y_mode=num_modes,
        z_min=z_min,
        z_max=z_max,
        ds_factor=cur_ds,
        use_layernorm=use_layernorm,
    ).to(device)

    optimizer = torch.optim.Adam(
        lr=1e-3,
        params=filter(lambda p: p.requires_grad, model.parameters()),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=0.1
    )

    t = tqdm.trange(num_epochs)
    for epoch in t:
        led_idices = list(np.arange(ID_len))  # list(np.random.permutation(ID_len)) #
        # _fill = len(led_idices) - (len(led_idices) % led_batch_size)
        # led_idices = led_idices + list(np.random.choice(led_idices, _fill, replace=False))
        dzs = torch.FloatTensor([0.0]).to(device)

        if use_c2f and c2f_sche[epoch] < model.ds_factor:
            model.init_scale_grids(ds_factor=c2f_sche[epoch])
            print(f"ds_factor changed to {c2f_sche[epoch]}")
            model_fn = torch.jit.trace(model, dzs[0:1])

        if epoch == 0:
            if is_os == "Windows":
                model_fn = torch.jit.trace(model, dzs[0:1])
            elif is_os == "Linux":
                model_fn = torch.compile(model, backend="inductor")
            else:
                raise NotImplementedError

        # TODO: Replace this and iterate over RGB wavelengths instead of defocus depth
        for dz in dzs:
            dz = dz.unsqueeze(0)

            for it in range(ID_len // led_batch_size):  # + 1
                model.zero_grad()
                dfmask = torch.exp(
                    1j
                    * kzz.repeat(dz.shape[0], 1, 1)
                    * dz[:, None, None].repeat(1, kzz.shape[1], kzz.shape[2])
                )
                led_num = led_idices[it * led_batch_size : (it + 1) * led_batch_size]
                dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
                spectrum_mask_ampli = Pupil0.repeat(
                    len(dz), len(led_num), 1, 1
                ) * torch.abs(dfmask)
                spectrum_mask_phase = Pupil0.repeat(len(dz), len(led_num), 1, 1) * (
                    torch.angle(dfmask) + 0
                )  # 0 represent Pupil0 Phase
                spectrum_mask = spectrum_mask_ampli * torch.exp(
                    1j * spectrum_mask_phase
                )

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    # TODO: Change model input to wavelength
                    img_ampli, img_phase = model_fn(dz)
                    img_complex = img_ampli * torch.exp(1j * img_phase)
                    # print("NN out shape", img_complex.shape)
                    uo, vo = ledpos_true[led_num, 0], ledpos_true[led_num, 1]
                    x_0, x_1 = vo - M // 2, vo + M // 2
                    y_0, y_1 = uo - N // 2, uo + N // 2

                    oI_cap = torch.sqrt(Isum[:, :, led_num])
                    oI_cap = (
                        oI_cap.permute(2, 0, 1).unsqueeze(0).repeat(len(dz), 1, 1, 1)
                    )
                    # print("measurement shape", oI_cap.shape)

                    oI_sub = get_sub_spectrum(
                        img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, MAGimg
                    )
                    # print("propagated shape", oI_sub.shape)
                    # exit()

                    l1_loss = F.smooth_l1_loss(oI_cap, oI_sub)
                    loss = l1_loss
                    mse_loss = F.mse_loss(oI_cap, oI_sub)

                loss.backward()

                psnr = 10 * -torch.log10(mse_loss).item()
                t.set_postfix(Loss=f"{loss.item():.4e}", PSNR=f"{psnr:.2f}")
                optimizer.step()

        scheduler.step()

        if (
            (epoch + 1) % 10 == 0
            or (epoch % 2 == 0 and epoch < 20)
            or epoch == num_epochs
        ):

            amplitude = (img_ampli[0].float()).cpu().detach().numpy()
            phase = (img_phase[0].float()).cpu().detach().numpy()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

            im = axs[0].imshow(amplitude, cmap="gray")
            axs[0].axis("image")
            axs[0].set_title("Reconstructed amplitude")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            im = axs[1].imshow(phase, cmap="gray")  # - phase.mean()
            axs[1].axis("image")
            axs[1].set_title("Reconstructed phase")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            plt.savefig(f"{vis_dir}/e_{epoch}.png")

    save_path = os.path.join("trained_models", sample + "_" + color + ".pth")
    save_model_with_required_grad(model, save_path)
