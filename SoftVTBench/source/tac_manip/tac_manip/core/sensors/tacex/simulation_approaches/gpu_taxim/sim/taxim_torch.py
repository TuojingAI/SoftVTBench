from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_scatter
from torch.nn.functional import conv2d
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from .calibration import CALIB_GELSIGHT
from .taxim_impl import ArrayType, TaximImpl


@torch.jit.script
def fast_conv2d(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Uses Fast-Fourier-Transformation to compute a 2D convolution efficiently. The speed-up is more significant for
    larger kernel sizes.
    :param input:  A (..., C, H, W) tensor containing the 2D data to compute the convolution for.
    :param kernel: A (KH, KW) tensor containing the kernel.
    :return: A (..., C, H, W) tensor containing the result.
    """
    assert all(e % 2 == 1 for e in kernel.shape[-2:])
    input_padded = input

    kpy = input_padded.shape[-2] - kernel.shape[-2]
    kpx = input_padded.shape[-1] - kernel.shape[-1]
    kernel_pad_size = (0, kpx, 0, kpy)
    kernel_padded = torch.nn.functional.pad(kernel, kernel_pad_size, mode="constant", value=0.0)

    input_ft = torch.fft.fft2(input_padded)
    kernel_ft = torch.fft.fft2(kernel_padded[None])
    output_ft = input_ft * torch.conj(kernel_ft)
    output = torch.real(torch.fft.ifft2(output_ft))[
        ...,
        : input.shape[-2] - (kernel.shape[-2] - 1),
        : input.shape[-1] - (kernel.shape[-1] - 1),
    ]
    return output


class TaximTorch(torch.nn.Module, TaximImpl[torch.Tensor, torch.device]):
    __constants__ = ["_device", "_shadow_depth_0", "_gel_map_shift"]

    def __init__(
        self,
        calib_folder: Path = CALIB_GELSIGHT,
        params: dict[str, dict[str, Any]] | None = None,
        device: torch.device | str = "cpu",
    ):
        """

        :param device:       Torch device on which the computation should run.
        :param calib_folder: Path to the folder with the calibration files.
        :param params:       Simulator parameters. Values set in this dictionary override values set in params.json
                             in the calib_folder.
        """
        with torch.no_grad():
            torch.nn.Module.__init__(self)
            TaximImpl.__init__(
                self,
                "torch",
                torch.device(device),
                calib_folder=calib_folder,
                params=params,
            )

            # Polynomial calibration file
            data = np.load(str(calib_folder / "polycalib.npz"))
            # The order of RGB in the data got mixed up: grad_b and grad_r are switched
            self.__poly_grad = (
                torch.from_numpy(np.stack([data["grad_b"], data["grad_g"], data["grad_r"]], axis=0) / 255)
                .float()
                .to(device)
            )

            gel_map = self.__np_img_to_torch(np.load(str(calib_folder / "gelmap.npy")), has_channels=False)
            gel_map = (
                self.__gaussian_blur(gel_map, self.sim_params.deform_final_sigma(gel_map.shape[1:]))[0]
                * self.sensor_params.pixmm
            )

            # Normalize gel map to have a maximum of 0
            self.__gel_map_shift = gel_map.max().item()
            self.__gel_map_full_res = gel_map - self.__gel_map_shift

            data_file = np.load(str(calib_folder / "dataPack.npz"), allow_pickle=True)
            f0 = self.__bgr_to_rgb(self.__np_img_to_torch(data_file["f0"] / 255))
            self.__bg_proc = self.__process_initial_frame(f0)

            # Shadow calibration
            self.__shadow_depth_0 = 0.4
            shadow_data = np.load(str(calib_folder / "shadowTable.npz"), allow_pickle=True)
            direction = torch.from_numpy(shadow_data["shadowDirections"]).float().to(device)

            # use a fan of angles around the direction
            fan_angle = self.sim_params.fan_angle
            num_fan_rays = int(fan_angle * 2 / self.sim_params.fan_precision)
            self.__shadow_direction_fan_angles = direction.unsqueeze(-1) + torch.linspace(
                -fan_angle, fan_angle, num_fan_rays, device=device
            )

            shadow_table = shadow_data["shadowTable"]
            # Append an extra empty entry for heights outside the range
            # Flipping here to transform BGR format to RGB
            shadow_table = np.concatenate(
                [
                    np.flip(shadow_table, axis=0),
                    [[[]] * shadow_table.shape[1]] * shadow_table.shape[0],
                ],
                axis=2,
            )
            max_shadow_table_len = max(map(len, shadow_table.reshape((-1,))))
            self.__shadow_table_padded = (
                torch.tensor(
                    [e + [np.inf] * (max_shadow_table_len - len(e)) for e in shadow_table.reshape((-1,))],
                    device=device,
                    dtype=torch.float,
                ).reshape(shadow_table.shape + (-1,))
                / 255
            )

            self.__get_gel_map_cached = lru_cache(self.__get_gel_map)
            self.__get_precomputed_features_cached = lru_cache(self.__get_precomputed_features)
            self.__get_background_img_cached = lru_cache(self.__get_background_img)

    @property
    def background_img(self):
        return self.__get_background_img_cached((self.height, self.width))

    def __get_background_img(self, shape: tuple[int, int]) -> torch.Tensor:
        return resize(self.__bg_proc, list(shape), interpolation=InterpolationMode.BILINEAR)

    def __get_precomputed_features(self, shape: tuple[int, int]) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0, self.height, shape[0] + 1, device=self.device)[:-1],
            torch.linspace(0, self.width, shape[1] + 1, device=self.device)[:-1],
            indexing="ij",
        )
        xf = xx.flatten()
        yf = yy.flatten()
        return torch.stack(
            [
                xf * xf,
                yf * yf,
                xf * yf,
                xf,
                yf,
                torch.ones(shape[0] * shape[1], device=self.device),
            ],
            dim=-1,
        )

    def __get_gel_map(self, shape: tuple[int, int]) -> torch.Tensor:
        return resize(
            self.__gel_map_full_res[None],
            list(shape),
            interpolation=InterpolationMode.BILINEAR,
        )[0]

    def convert_height_map(self, height_map: np.ndarray) -> ArrayType:
        return torch.from_numpy(height_map).to(self.device).float()

    def img_to_numpy(self, img: ArrayType) -> np.ndarray:
        b_dims = len(img.shape[:-3])
        return img.permute(*range(b_dims), -2, -1, -3).cpu().numpy()

    @torch.jit.export
    def _render_impl(
        self,
        height_map: ArrayType,
        with_shadow: bool = True,
        press_depth: float | None = None,
        orig_hm_fmt: bool = False,
    ) -> ArrayType:
        with torch.no_grad():
            batch_shape = height_map.shape[:-2]
            height_map = height_map.reshape((-1,) + height_map.shape[-2:])

            if orig_hm_fmt:
                height_map = self.__gel_map_shift - height_map

            if press_depth is not None:
                height_map = self.__get_shifted_height_map(press_depth, height_map)

            # simulate tactile images
            sim_img = self.__render(height_map, shadow=with_shadow)

            sim_img = sim_img.reshape(batch_shape + sim_img.shape[-3:])
            return sim_img

    @torch.jit.ignore
    def render(
        self,
        height_map: torch.Tensor,
        with_shadow: bool = True,
        press_depth: float | None = None,
        orig_hm_fmt: bool = False,
    ) -> torch.Tensor:
        return super().render(height_map, with_shadow, press_depth, orig_hm_fmt)

    def forward(
        self,
        height_map: torch.Tensor,
        with_shadow: bool = False,
        press_depth: float | None = None,
        orig_hm_fmt: bool = False,
    ) -> torch.Tensor:
        return self.render(height_map, with_shadow, press_depth, orig_hm_fmt)

    def __bgr_to_rgb(self, img: torch.Tensor) -> torch.Tensor:
        """
        Transforms an image from BGR to RGB.
        :param img: Image to transform. Assumed to have shape (..., 3, H, W), where H and W are the height and width of
                    the image.
        :return: Transformed image
        """
        return torch.flip(img, dims=(-3,))

    def __render(self, height_map: torch.Tensor, shadow: bool = False) -> torch.Tensor:
        """
        Simulates the tactile image from the height map.
        :param height_map: Height map to generate tactile image for in mm.
        :param shadow:     Whether to generate shadows.
        :return: A 3xHxW torch tensor containing the resulting image in RGB format.
        """
        shape = height_map.shape[1:]
        height, width = shape

        deformed_gel, contact_mask = self.__compute_gel_pad_deformation(height_map)

        # generate gradients of the height map
        deformed_gel_px = deformed_gel / self.sensor_params.pixmm
        grad_mag, grad_dir = self.__generate_normals(-deformed_gel_px)

        # generate raw simulated image without background
        # discretize grids
        x_binr = 0.5 * torch.pi / (self.sensor_params.num_bins - 1)  # x [0,pi/2]
        y_binr = 2 * torch.pi / (self.sensor_params.num_bins - 1)  # y [-pi, pi]

        idx_mag = torch.floor(grad_mag / x_binr).long()
        idx_dir = torch.floor((grad_dir + torch.pi) / y_binr).long()

        # look up polynomial table and assign intensity
        params = self.__poly_grad[:, idx_mag, idx_dir].transpose(0, 1)
        params_flat = torch.flatten(params, start_dim=-3, end_dim=-2)
        sim_img_flat = (self.__get_precomputed_features_cached(shape).unsqueeze(0) * params_flat).sum(-1)
        sim_img_r = torch.unflatten(sim_img_flat, dim=-1, sizes=(height, width))

        if not shadow:
            # Add background to simulated image
            sim_img = sim_img_r + self.__get_background_img_cached(shape)
            return torch.clip(sim_img, 0, 1)

        # find shadow attachment area
        kernel_size_float = np.array(self.sim_params.shadow_attachment_kernel_size(shape))
        kernel_size_float_total = np.round(kernel_size_float * 2).astype(np.int_)
        first_round = kernel_size_float_total // 2
        rounds = [first_round, kernel_size_float_total - first_round]
        dilated_mask = contact_mask[..., None, :, :].float()
        for ks in rounds:
            kernel = torch.ones((1, 1, *np.flip(np.maximum(1, ks))), device=self.device)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                dilated_mask = conv2d(dilated_mask, kernel, padding="same")
        enlarged_mask = dilated_mask[..., 0, :, :] != 0
        boundary_contact_mask = enlarged_mask & ~contact_mask

        # get normal index to shadow table
        norm_map = grad_dir[boundary_contact_mask] + np.pi
        norm_idx = torch.floor(norm_map / self.sim_params.discretize_precision).long()

        # get height index to shadow table
        contact_height = self.__get_gel_map_cached(shape) - deformed_gel
        contact_height_px = contact_height / self.sensor_params.pixmm
        contact_map = contact_height_px[boundary_contact_mask]
        height_idx = torch.floor(
            (contact_map * self.sensor_params.pixmm - self.__shadow_depth_0) / self.sim_params.height_precision
        ).long()
        height_idx_shifted = height_idx + 6
        max_height_idx = self.__shadow_table_padded.shape[2] - 1
        height_idx_shifted[(height_idx_shifted < 0) | (height_idx_shifted >= max_height_idx)] = max_height_idx

        shadow_table_sel = self.__shadow_table_padded[:, norm_idx, height_idx_shifted]
        thetas = self.__shadow_direction_fan_angles[norm_idx]
        steps = torch.arange(shadow_table_sel.shape[-1], device=self.device)

        # (x,y) coordinates of all pixels to attach shadow
        batch_idx, y_coord, x_coord = torch.where(boundary_contact_mask)
        x_coord_r = x_coord.reshape((-1, 1, 1))
        y_coord_r = y_coord.reshape((-1, 1, 1))

        shadow_step = self.sim_params.shadow_step(shape)

        shadow_coords_x = (
            x_coord_r + shadow_step[1] * (steps.reshape((1, 1, -1)) + 1) * torch.cos(thetas).unsqueeze(-1)
        ).long()
        shadow_coords_y = (
            y_coord_r + shadow_step[0] * (steps.reshape((1, 1, -1)) + 1) * torch.sin(thetas).unsqueeze(-1)
        ).long()
        sc_x_clipped = torch.clip(shadow_coords_x, 0, width - 1)
        sc_y_clipped = torch.clip(shadow_coords_y, 0, height - 1)
        sc_valid = (
            (shadow_coords_x >= 0)
            & (shadow_coords_x < width)
            & (shadow_coords_y >= 0)
            & (shadow_coords_y < height)
            & (
                deformed_gel_px[batch_idx[..., None, None], y_coord_r, x_coord_r]
                < deformed_gel_px[batch_idx[..., None, None], sc_y_clipped, sc_x_clipped]
            )
        )
        valid_sc_x = shadow_coords_x[sc_valid]
        valid_sc_y = shadow_coords_y[sc_valid]
        valid_sc_batch = torch.broadcast_to(batch_idx[:, None, None], sc_valid.shape)[sc_valid]
        sc_valid_idx_coord, _, sc_valid_idx_shadow_coord = torch.where(sc_valid)
        valid_shadow_coords_values = shadow_table_sel[:, sc_valid_idx_coord, sc_valid_idx_shadow_coord]
        valid_shadow_coords_flat = (valid_sc_batch * height + valid_sc_y) * width + valid_sc_x
        shadow_img_flat = torch.full(
            (sim_img_r.shape[1], sim_img_r.shape[0] * width * height),
            np.inf,
            device=self.device,
        )
        bs = sim_img_r.shape[0]
        torch_scatter.scatter_min(
            valid_shadow_coords_values,
            valid_shadow_coords_flat,
            dim_size=bs * width * height,
            out=shadow_img_flat,
        )
        shadow_img = torch.reshape(shadow_img_flat, (sim_img_r.shape[1], bs, height, width)).transpose(0, 1)
        sim_img_r = torch.minimum(sim_img_r, shadow_img)

        shadow_sim = self.__gaussian_blur(
            sim_img_r,
            self.sim_params.shadow_blur_sigma(shape),
        )
        shadow_sim_img = shadow_sim + self.__get_background_img_cached(shape)
        shadow_sim_img_blurred = self.__gaussian_blur(shadow_sim_img, self.sim_params.deform_final_sigma(shape))

        return torch.clip(shadow_sim_img_blurred, 0, 1)

    def __np_img_to_torch(self, img: np.ndarray, has_channels: bool = True) -> torch.Tensor:
        """
        Converts a numpy image to torch.
        :param img: Image to convert.
        :param has_channels: Whether the image has a channel dimension.
        :return: Torch tensor containing the image.
        """
        img_torch = torch.from_numpy(img).to(self.device).float()
        if has_channels:
            b_dims = len(img_torch.shape[:-3])
            return img_torch.permute(*range(b_dims), -1, -3, -2)
        else:
            return img_torch[..., None, :, :]

    @classmethod
    def __get_gaussian_kernel1d(cls, sigma: float, kernel_size: int) -> torch.Tensor:
        x = torch.linspace(-(kernel_size - 1) * 0.5, (kernel_size - 1) * 0.5, steps=kernel_size)
        pdf = torch.exp(-0.5 * (x / sigma).pow(2))
        return pdf / pdf.sum()

    @classmethod
    def __get_gaussian_kernel2d(
        cls,
        sigma: list[float],
        kernel_size: list[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        kernel1d_x = cls.__get_gaussian_kernel1d(sigma[0], kernel_size[0]).to(device, dtype=dtype)
        kernel1d_y = cls.__get_gaussian_kernel1d(sigma[1], kernel_size[1]).to(device, dtype=dtype)
        kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
        return kernel2d

    @classmethod
    def __gaussian_blur(
        cls,
        img: torch.Tensor,
        sigma: float,
        kernel_size: int | None = None,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur to the given image.
        :param img:         A 3xHxW torch tensor containing the image.
        :param sigma:       Standard deviation used for the blurring.
        :param kernel_size: Kernel size used for the blurring. If not given, it is computed such that the weight of the
                            outermost pixel is less than 1e-5.
        :return: A 3xHxW torch tensor containing the blurred image.
        """
        if kernel_size is None:
            eps = 1e-5
            sigma_np = np.array(sigma)
            # Ensure kernel size is odd
            kernel_size = (
                np.round(np.sqrt(-2 * np.log(eps * np.sqrt(2 * np.pi) * sigma_np)) * sigma_np).astype(np.int_) // 2 * 2
                + 1
            ).tolist()
        kernel = cls.__get_gaussian_kernel2d(
            sigma,
            kernel_size,
            dtype=torch.float,
            device=img.device,
        )
        p = (np.array(kernel_size) - 1) // 2
        img_padded = torch.nn.functional.pad(img, (p[0], p[0], p[1], p[1]), mode="reflect")
        return fast_conv2d(img_padded, kernel)

    def __process_initial_frame(self, f0: torch.Tensor):
        """
        Conduct some preprocessing on the initial frame.
        :param f0: A 3xHxW torch tensor containing the initial frame.
        :return: A 3xHxW torch tensor containing the processed initial frame.
        """
        # gaussian filtering with square kernel
        f0_blurred = self.__gaussian_blur(f0, self.sim_params.initial_frame_sigma(f0.shape[1:]))

        # Checking the difference between original and filtered image
        d_i = torch.mean(f0_blurred - f0, dim=0)

        # Mixing image based on the difference between original and filtered image
        fmp = self.sim_params.frame_mixing_percentage
        thresh = self.sim_params.diff_threshold

        return torch.where((d_i < thresh).unsqueeze(0), fmp * f0_blurred + (1 - fmp) * f0, f0)

    def __get_shifted_height_map(self, pressing_depth_mm: float, height_map: torch.Tensor) -> torch.Tensor:
        """
        Generate the shifted height map of the gel surface by interacting the object with the gel pad model. After
        shifting, the closest point of the height map will be pressing_depth_mm below the furthest point of the gel.
        :param pressing_depth_mm: How deep the object is pressed into the gel in mm.
        :param height_map:        Height map of the object in mm.
        :return: Shifted height map in mm.
        """
        # shift the height map to interact with the object
        return height_map - height_map.amin(-1, keepdim=True).amin(-2, keepdim=True) - pressing_depth_mm.view(-1, 1, 1)

    def __compute_gel_pad_deformation(self, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the deformation of the gel pad.
        :param height_map: Height map of the object in mm.
        :return: Height map of the deformed gel pad in mm.
        """
        pressing_depth_mm = -height_map.amin(-1).amin(-1)

        # get the contact area
        contact_mask = height_map < 0

        joined_height_map = torch.minimum(height_map, self.__get_gel_map_cached(height_map.shape[1:]))

        # contact mask which is a little smaller than the real contact mask
        mask = torch.logical_and(
            joined_height_map - self.__get_gel_map_cached(height_map.shape[1:])
            < -pressing_depth_mm[..., None, None] * self.sim_params.contact_scale,
            contact_mask,
        )

        # approximate soft body deformation with pyramid gaussian_filter
        height_map_blurred = joined_height_map
        for sigma in zip(*self.sim_params.deform_pyramid_sigma(height_map.shape[1:])):
            height_map_blurred = self.__gaussian_blur(height_map_blurred.unsqueeze(0), sigma)[0]
            height_map_blurred[mask] = joined_height_map[mask]
        height_map_blurred = self.__gaussian_blur(
            height_map_blurred.unsqueeze(0),
            self.sim_params.deform_final_sigma(height_map.shape[1:]),
        )[0]

        return height_map_blurred, mask

    def __generate_normals(self, height_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the gradient (magnitude & direction) map from the height map.
        :param height_map: Height map to compute the gradients for.
        :return: A tuple containing (1) the magnitudes and (2) the directions of the gradients.
        """
        h, w = height_map.shape[-2:]
        top = height_map[..., 0 : h - 2, 1 : w - 1]  # z(x-1,y)
        bot = height_map[..., 2:h, 1 : w - 1]  # z(x+1,y)
        left = height_map[..., 1 : h - 1, 0 : w - 2]  # z(x,y-1)
        right = height_map[..., 1 : h - 1, 2:w]  # z(x,y+1)
        dzdx = (bot - top) / 2.0
        dzdy = (right - left) / 2.0

        dzdx_norm = dzdx * height_map.shape[1] / self.height
        dzdy_norm = dzdy * height_map.shape[2] / self.width

        mag_tan = torch.sqrt(dzdx_norm**2 + dzdy_norm**2)
        grad_mag = torch.arctan(mag_tan)
        valid_mask = mag_tan != 0
        grad_dir = torch.zeros(mag_tan.shape[:-2] + (h - 2, w - 2), device=self.device)
        grad_dir[valid_mask] = torch.arctan2(
            dzdx_norm[valid_mask] / mag_tan[valid_mask],
            dzdy_norm[valid_mask] / mag_tan[valid_mask],
        )

        grad_mag = torch.nn.functional.pad(grad_mag, (1, 1, 1, 1), "replicate")
        grad_dir = torch.nn.functional.pad(grad_dir, (1, 1, 1, 1), "replicate")
        return grad_mag, grad_dir
