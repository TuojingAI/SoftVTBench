from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jaxlib.xla_extension
import numpy as np

from .calibration import CALIB_GELSIGHT
from .taxim_impl import TaximImpl


class TaximJax(TaximImpl[jax.Array, jaxlib.xla_extension.Device | None]):
    def __init__(
        self,
        calib_folder: Path = CALIB_GELSIGHT,
        params: dict[str, dict[str, Any]] | None = None,
        device: jaxlib.xla_extension.Device | None = None,
        shadow_method: Literal["constant_time", "fast"] = "fast",
        shadow_computation_chunk_size: int = 4096,
    ):
        """

        :param calib_folder:                    Path to the folder with the calibration files.
        :param params:                          Simulator parameters. Values set in this dictionary override values set
                                                in params.json in the calib_folder.
        :param device:                          Device to use. If None, the default device is used.
        :param shadow_method:                   Which method to use for the shadow computation. The constant_time
                                                implementation is slower on average than the fast implementation, but
                                                its runtime is independent of the number of contact pixels in the image.
        :param shadow_computation_chunk_size:   Chunk size to use for the fast shadow computation. This is only used if
                                                use_constant_time_impl is False.
        """
        super().__init__("jax", device, calib_folder=calib_folder, params=params)
        # Polynomial calibration file
        data = np.load(str(calib_folder / "polycalib.npz"))
        # The order of RGB in the data got mixed up: grad_b and grad_r are switched
        self.__poly_grad = jnp.stack([data["grad_b"], data["grad_g"], data["grad_r"]], axis=-1) / 255

        gel_map = jnp.load(str(calib_folder / "gelmap.npy"))
        gel_map = (
            self.__gaussian_blur(
                gel_map,
                self.sim_params.deform_final_sigma(gel_map.shape),
                has_channel_dim=False,
            )
            * self.sensor_params.pixmm
        )

        # Normalize gel map to have a maximum of 0
        self.__gel_map_shift = gel_map.max().item()
        self.__gel_map_full_res = gel_map - self.__gel_map_shift

        data_file = np.load(str(calib_folder / "dataPack.npz"), allow_pickle=True)
        f0 = self.__bgr_to_rgb(data_file["f0"] / 255)
        self.__bg_proc = self.__process_initial_frame(f0)

        # Shadow calibration
        self.__shadow_depth_0 = 0.4
        shadow_data = np.load(str(calib_folder / "shadowTable.npz"), allow_pickle=True)
        direction = jnp.array(shadow_data["shadowDirections"])

        # use a fan of angles around the direction
        fan_angle = self.sim_params.fan_angle
        num_fan_rays = int(fan_angle * 2 / self.sim_params.fan_precision)
        self.__shadow_direction_fan_angles = direction[..., None] + jnp.linspace(-fan_angle, fan_angle, num_fan_rays)

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
        shadow_table_padded = (
            jnp.array(
                [e + [jnp.inf] * (max_shadow_table_len - len(e)) for e in shadow_table.reshape((-1,))],
            ).reshape(shadow_table.shape + (-1,))
            / 255
        )
        self.__shadow_table_padded = shadow_table_padded.transpose(1, 2, 3, 0)

        self.__render_fn = jax.jit(
            self.__render_impl_jitable,
            static_argnames=["with_shadow", "orig_hm_fmt"],
            device=device,
        )
        self.__shadow_method = shadow_method
        self.__shadow_computation_chunk_size = shadow_computation_chunk_size

    def __get_background_img(self, shape: tuple[int, int]) -> jax.Array:
        return jax.image.resize(self.__bg_proc, shape + (3,), method="linear")

    def __get_gel_map(self, shape: tuple[int, int]) -> jax.Array:
        return jax.image.resize(self.__gel_map_full_res, shape, method="linear")

    def _render_impl(
        self,
        height_map: jax.Array,
        with_shadow: bool = True,
        press_depth: float | None = None,
        orig_hm_fmt: bool = False,
    ) -> jax.Array:
        return self.__render_fn(
            height_map,
            with_shadow,
            jnp.nan if press_depth is None else press_depth,
            orig_hm_fmt,
        )

    def img_to_numpy(self, img: jax.Array) -> np.ndarray:
        return np.array(img)

    def convert_height_map(self, height_map: np.ndarray) -> np.ndarray:
        return height_map

    def __render_impl_jitable(
        self,
        height_map: jax.Array,
        with_shadow: bool,
        press_depth: float,
        orig_hm_fmt: bool,
    ) -> jax.Array:
        batch_shape = height_map.shape[:-2]
        height_map = height_map.reshape((-1,) + height_map.shape[-2:])

        if orig_hm_fmt:
            height_map = self.__gel_map_shift - height_map

        height_map = jax.lax.cond(
            jnp.isnan(press_depth),
            lambda: height_map,
            lambda: self.__get_shifted_height_map(press_depth, height_map),
        )

        # simulate tactile images
        sim_img = jax.vmap(lambda hm: self.__render(hm, shadow=with_shadow))(height_map)

        sim_img = sim_img.reshape(batch_shape + sim_img.shape[-3:])
        return sim_img

    def __bgr_to_rgb(self, img: jax.Array) -> jax.Array:
        """
        Transforms an image from BGR to RGB.
        :param img: Image to transform. Assumed to have shape (..., 3, H, W), where H and W are the height and width of
                    the image.
        :return: Transformed image
        """
        return jnp.flip(img, axis=-1)

    def __render(self, height_map: jax.Array, shadow: bool = False) -> jax.Array:
        """
        Simulates the tactile image from the height map.
        :param height_map: Height map to generate tactile image for in mm.
        :param shadow:     Whether to generate shadows.
        :return: A HxWx3 array containing the resulting image in RGB format.
        """
        shape = height_map.shape
        height, width = shape

        deformed_gel, contact_mask = self.__compute_gel_pad_deformation(height_map)

        # generate gradients of the height map
        deformed_gel_px = deformed_gel / self.sensor_params.pixmm
        grad_mag, grad_dir = self.__generate_normals(-deformed_gel_px)

        # generate raw simulated image without background
        # discretize grids
        x_binr = 0.5 * jnp.pi / (self.sensor_params.num_bins - 1)  # x [0,pi/2]
        y_binr = 2 * jnp.pi / (self.sensor_params.num_bins - 1)  # y [-pi, pi]

        idx_mag = jnp.floor(grad_mag / x_binr).astype(jnp.int32)
        idx_dir = jnp.floor((grad_dir + jnp.pi) / y_binr).astype(jnp.int32)

        # look up polynomial table and assign intensity
        params = self.__poly_grad[idx_mag, idx_dir]
        params_flat = params.reshape((-1, *params.shape[-2:]))
        yy, xx = jnp.meshgrid(
            jnp.linspace(0, self.height, shape[0], endpoint=False),
            jnp.linspace(0, self.width, shape[1], endpoint=False),
            indexing="ij",
        )
        xf = xx.flatten()
        yf = yy.flatten()
        features = jnp.stack(
            [xf * xf, yf * yf, xf * yf, xf, yf, jnp.ones(shape[0] * shape[1])],
            axis=-1,
        )

        sim_img_flat = (features[..., None] * params_flat).sum(-2)
        sim_img_r = sim_img_flat.reshape((*shape, params.shape[-1]))

        if not shadow:
            # Add background to simulated image
            sim_img = sim_img_r + self.__get_background_img(shape)
            return jnp.clip(sim_img, 0, 1)

        # find shadow attachment area
        kernel_size_float = np.array(self.sim_params.shadow_attachment_kernel_size(shape))
        kernel_size_float_total = np.round(kernel_size_float * 2).astype(np.int_)
        first_round = kernel_size_float_total // 2
        rounds = [first_round, kernel_size_float_total - first_round]
        dilated_mask = contact_mask.astype(jnp.float32)
        for ks in rounds:
            kernel = jnp.ones(tuple(np.flip(np.maximum(1, ks))))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                dilated_mask = jax.scipy.signal.convolve(dilated_mask, kernel, mode="same")
        enlarged_mask = dilated_mask != 0
        boundary_contact_mask = enlarged_mask & ~contact_mask

        # get normal index to shadow table
        norm_map = grad_dir + np.pi
        norm_idx = jnp.floor(norm_map / self.sim_params.discretize_precision).astype(jnp.int32)

        # get height index to shadow table
        contact_height = self.__get_gel_map(shape) - deformed_gel
        contact_height_px = contact_height / self.sensor_params.pixmm
        height_idx = jnp.floor(
            (contact_height_px * self.sensor_params.pixmm - self.__shadow_depth_0) / self.sim_params.height_precision
        ).astype(jnp.int32)
        height_idx_shifted = height_idx + 6
        max_height_idx = self.__shadow_table_padded.shape[1] - 1
        height_idx_shifted = jnp.where(
            (height_idx_shifted < 0) | (height_idx_shifted >= max_height_idx),
            max_height_idx,
            height_idx_shifted,
        )

        shadow_table_sel = self.__shadow_table_padded[norm_idx, height_idx_shifted]
        thetas = self.__shadow_direction_fan_angles[norm_idx]
        step_count = shadow_table_sel.shape[-2]

        # (x,y) coordinates of all pixels to attach shadow
        y_coord, x_coord = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")

        chunk_size = self.__shadow_computation_chunk_size
        output_size = int(math.ceil(height * width / chunk_size)) * chunk_size
        valid_coords_y, valid_coords_x = jnp.where(boundary_contact_mask, size=output_size, fill_value=-1)

        def cast_shadows(steps: jax.Array, coords_y: jax.Array, coords_x: jax.Array, img: jax.Array) -> jax.Array:
            shadow_step = self.sim_params.shadow_step(shape)
            th = thetas[coords_y, coords_x]
            shadow_coords_x = (
                coords_x[..., None, None] + shadow_step[1] * (steps + 1) * jnp.cos(th)[..., None]
            ).astype(jnp.int32)
            shadow_coords_y = (
                coords_y[..., None, None] + shadow_step[0] * (steps + 1) * jnp.sin(th)[..., None]
            ).astype(jnp.int32)
            shadow_coords_in_bounds = (
                (shadow_coords_x >= 0) & (shadow_coords_x < width) & (shadow_coords_y >= 0) & (shadow_coords_y < height)
            )
            sc_x_clipped = jnp.clip(shadow_coords_x, 0, width - 1)
            sc_y_clipped = jnp.clip(shadow_coords_y, 0, height - 1)
            sc_valid = (
                shadow_coords_in_bounds
                & boundary_contact_mask[coords_y, coords_x, None, None]
                & (deformed_gel_px[coords_y, coords_x, None, None] < deformed_gel_px[sc_y_clipped, sc_x_clipped])
            )
            sc_values = jnp.where(
                sc_valid[..., None],
                shadow_table_sel[coords_y[..., None, None], coords_x[..., None, None], steps, :],
                jnp.inf,
            )
            return img.at[sc_y_clipped, sc_x_clipped].min(sc_values)

        def fast_cond_fn(args: tuple[int, jax.Array]) -> bool:
            i, _ = args
            # Checking the first element is sufficient
            return (i < height * width) & (valid_coords_y[i] != -1)

        def fast_loop_body_fn(args: tuple[int, jax.Array]) -> tuple[int, jax.Array]:
            i, img = args
            vcy = jax.lax.dynamic_slice(valid_coords_y, (i,), (chunk_size,))
            vcx = jax.lax.dynamic_slice(valid_coords_x, (i,), (chunk_size,))
            img = cast_shadows(jnp.arange(step_count), vcy, vcx, img)
            return i + chunk_size, img

        # Compute this in a loop to reduce the memory footprint
        def constant_time_loop_body_fn(i: int, img: jax.Array) -> jax.Array:
            return cast_shadows(
                jnp.reshape(i, (1,)),
                y_coord.reshape((-1,)),
                x_coord.reshape((-1,)),
                img,
            )

        if self.__shadow_method == "fast":
            sim_img_r = jax.lax.while_loop(fast_cond_fn, fast_loop_body_fn, (0, sim_img_r))[1]
        else:
            sim_img_r = jax.lax.fori_loop(
                0,
                shape[1] * shape[0],
                constant_time_loop_body_fn,
                sim_img_r,
            )

        shadow_sim = self.__gaussian_blur(
            sim_img_r,
            self.sim_params.shadow_blur_sigma(shape),
        )
        shadow_sim_img = shadow_sim + self.__get_background_img(shape)
        shadow_sim_img_blurred = self.__gaussian_blur(shadow_sim_img, self.sim_params.deform_final_sigma(shape))

        return jnp.clip(shadow_sim_img_blurred, 0, 1)

    @classmethod
    def __get_gaussian_kernel1d(cls, sigma: float, kernel_size: int) -> jax.Array:
        x = jnp.linspace(-(kernel_size - 1) * 0.5, (kernel_size - 1) * 0.5, num=kernel_size)
        pdf = jnp.exp(-0.5 * (x / sigma) ** 2)
        return pdf / pdf.sum()

    @classmethod
    def __get_gaussian_kernel2d(cls, kernel_size: tuple[int, int], sigma: tuple[float, float]) -> jax.Array:
        kernel1d_x = cls.__get_gaussian_kernel1d(sigma[0], kernel_size[0])
        kernel1d_y = cls.__get_gaussian_kernel1d(sigma[1], kernel_size[1])
        kernel2d = kernel1d_y[:, None] @ kernel1d_x[None, :]
        return kernel2d

    @classmethod
    def __gaussian_blur_single(
        cls,
        img: jax.Array,
        sigma: tuple[float, float],
        kernel_size: tuple[int, int] | None = None,
    ) -> jax.Array:
        if kernel_size is None:
            eps = 1e-5
            sigma_np = np.array(sigma)
            # Ensure kernel size is odd
            kernel_size = (
                np.round(np.sqrt(-2 * np.log(eps * np.sqrt(2 * np.pi) * sigma_np)) * sigma_np).astype(np.int_) // 2 * 2
                + 1
            ).tolist()
        kernel = cls.__get_gaussian_kernel2d(kernel_size, sigma)
        p = (np.array(kernel_size) - 1) // 2
        pad_width = ((p[1], p[1]), (p[0], p[0]))
        if len(img.shape) == 3:
            kernel = kernel[..., None]
            pad_width += ((0, 0),)
        img_padded = jnp.pad(img, pad_width, mode="reflect")
        if np.any(np.array(kernel_size) >= 8):
            method = "fft"
        else:
            method = "direct"
        return jax.scipy.signal.convolve(img_padded, kernel, mode="valid", method=method)

    @classmethod
    def __gaussian_blur(
        cls,
        img: jax.Array,
        sigma: float,
        kernel_size: int | None = None,
        has_channel_dim: bool = True,
    ) -> jax.Array:
        """
        Apply Gaussian blur to the given image.
        :param img:         A 3xHxW torch tensor containing the image.
        :param sigma:       Standard deviation used for the blurring.
        :param kernel_size: Kernel size used for the blurring. If not given, it is computed such that the weight of the
                            outermost pixel is less than 1e-5.
        :return: A HxWx3 array containing the blurred image.
        """
        if len(img.shape) >= 3 + int(has_channel_dim):
            return jax.vmap(lambda i: cls.__gaussian_blur_single(i, sigma, kernel_size))(img)
        return cls.__gaussian_blur_single(img, sigma, kernel_size)

    def __process_initial_frame(self, f0: jax.Array):
        """
        Conduct some preprocessing on the initial frame.
        :param f0: A HxWx3 array containing the initial frame.
        :return: A HxWx3 array containing the processed initial frame.
        """
        # gaussian filtering with square kernel
        f0_blurred = self.__gaussian_blur(f0, self.sim_params.initial_frame_sigma(f0.shape[:2]))

        # Checking the difference between original and filtered image
        d_i = jnp.mean(f0_blurred - f0, axis=0)

        # Mixing image based on the difference between original and filtered image
        fmp = self.sim_params.frame_mixing_percentage
        thresh = self.sim_params.diff_threshold

        return jnp.where((d_i < thresh)[0], fmp * f0_blurred + (1 - fmp) * f0, f0)

    def __get_shifted_height_map(self, pressing_depth_mm: float, height_map: jax.Array) -> jax.Array:
        """
        Generate the shifted height map of the gel surface by interacting the object with the gel pad model. After
        shifting, the closest point of the height map will be pressing_depth_mm below the furthest point of the gel.
        :param pressing_depth_mm: How deep the object is pressed into the gel in mm.
        :param height_map:        Height map of the object in mm.
        :return: Shifted height map in mm.
        """
        # shift the height map to interact with the object
        return height_map - height_map.min(axis=(-2, -1), keepdims=True) - pressing_depth_mm

    def __compute_gel_pad_deformation(self, height_map: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Compute the deformation of the gel pad.
        :param height_map: Height map of the object in mm.
        :return: Height map of the deformed gel pad in mm.
        """
        pressing_depth_mm = -height_map.min(axis=(-2, -1), keepdims=True)

        # get the contact area
        contact_mask = height_map < 0

        gel_map = self.__get_gel_map(height_map.shape)
        joined_height_map = jnp.minimum(height_map, gel_map)

        # contact mask which is a little smaller than the real contact mask
        mask = (joined_height_map - gel_map < -pressing_depth_mm * self.sim_params.contact_scale) & contact_mask

        # approximate soft body deformation with pyramid gaussian_filter
        height_map_blurred = joined_height_map
        for sigma in zip(*self.sim_params.deform_pyramid_sigma(height_map.shape)):
            height_map_blurred = self.__gaussian_blur(
                height_map_blurred,
                sigma,
                has_channel_dim=False,
            )
            height_map_blurred = jnp.where(mask, joined_height_map, height_map_blurred)
        height_map_blurred = self.__gaussian_blur(
            height_map_blurred,
            self.sim_params.deform_final_sigma(height_map.shape),
            has_channel_dim=False,
        )

        return height_map_blurred, mask

    def __generate_normals(self, height_map: jax.Array) -> tuple[jax.Array, jax.Array]:
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

        dzdx_norm = dzdx * height_map.shape[0] / self.height
        dzdy_norm = dzdy * height_map.shape[1] / self.width

        mag_tan = jnp.sqrt(dzdx_norm**2 + dzdy_norm**2)
        grad_mag = jnp.arctan(mag_tan)
        grad_dir = jnp.zeros(mag_tan.shape[:-2] + (h - 2, w - 2))
        grad_dir = jnp.where(
            mag_tan != 0,
            jnp.arctan2(dzdx_norm / mag_tan, dzdy_norm / mag_tan),
            grad_dir,
        )

        grad_mag = jnp.pad(grad_mag, ((1, 1), (1, 1)), "edge")
        grad_dir = jnp.pad(grad_dir, ((1, 1), (1, 1)), "edge")
        return grad_mag, grad_dir
