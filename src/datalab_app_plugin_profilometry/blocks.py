"""
Profiling data blocks for surface profilometry visualization.
"""

import time
import warnings
from pathlib import Path

import bokeh.embed
import numpy as np
from bokeh.layouts import column
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.plotting import figure
from pydatalab.blocks.base import DataBlock
from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME
from pydatalab.file_utils import get_file_info_by_id
from pydatalab.logger import LOGGER

from .wyko_reader import load_wyko_asc_cached


def downsample_image_block_average(
    image: np.ndarray, max_dimension: int = 2000
) -> tuple[np.ndarray, float]:
    """
    Downsample an image using block averaging to a target maximum dimension.

    This function preserves data characteristics better than decimation while
    reducing data size for visualization. NaN values are handled properly.

    Args:
        image: 2D numpy array to downsample
        max_dimension: Maximum size for the larger dimension (default: 2000px)

    Returns:
        tuple: (downsampled_image, downsample_factor)
            - downsampled_image: Downsampled 2D array
            - downsample_factor: Factor by which the image was downsampled
    """
    n_rows, n_cols = image.shape

    # Calculate downsampling factor based on larger dimension
    max_current = max(n_rows, n_cols)
    if max_current <= max_dimension:
        # No downsampling needed
        return image, 1.0

    downsample_factor = max_current / max_dimension
    LOGGER.debug(f"Downsampling factor calculated: {downsample_factor:.2f}")
    block_size = int(np.ceil(downsample_factor))

    # Calculate new dimensions
    new_rows = n_rows // block_size
    new_cols = n_cols // block_size

    # Trim image to be evenly divisible by block_size
    trimmed_rows = new_rows * block_size
    trimmed_cols = new_cols * block_size
    trimmed_image = image[:trimmed_rows, :trimmed_cols]

    # Reshape into blocks and compute nanmean along block dimensions
    # Shape: (new_rows, block_size, new_cols, block_size)
    blocks = trimmed_image.reshape(new_rows, block_size, new_cols, block_size)

    # Average over the block dimensions (axis 1 and 3)
    # Use nanmean to handle NaN values properly
    # Suppress "mean of empty slice" warning for blocks that are entirely NaN
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        downsampled = np.nanmean(blocks, axis=(1, 3))

    LOGGER.debug(
        f"Downsampled from {n_rows}x{n_cols} to {new_rows}x{new_cols} "
        f"(factor: {downsample_factor:.2f}x, block size: {block_size}x{block_size})"
    )

    return downsampled.astype(np.float32), downsample_factor


class ProfilingBlock(DataBlock):
    """Block for visualizing surface profilometry data."""

    accepted_file_extensions: tuple[str, ...] = (".asc",)
    blocktype = "profiling"
    name = "Surface Profiling"
    description = "This block can plot surface profilometry data from Wyko .ASC files"
    downsample_image_size: int = 3000  # Max dimension for image downsampling

    @property
    def plot_functions(self):
        return (self.generate_profiling_plot,)

    @staticmethod
    def _create_image_plot(
        image_data: np.ndarray,
        pixel_size: float | None = None,
        title: str = "Surface Profile",
        colorbar_label: str = "Height",
        cached_percentiles: tuple[float, float] | None = None,
        downsample_factor: float = 1.0,
    ):
        """
        Create a 2D Bokeh image plot for surface profilometry data.

        Args:
            image_data: 2D numpy array of height/intensity values (already downsampled if needed)
            pixel_size: Physical pixel size in mm (for axis scaling)
            title: Plot title
            colorbar_label: Label for the colorbar
            cached_percentiles: Optional tuple of (p1, p99) percentiles from cache
            downsample_factor: Factor by which the image was downsampled (for pixel size adjustment)

        Returns:
            Bokeh figure with image plot
        """
        t_start = time.perf_counter()

        # Get dimensions
        n_rows, n_cols = image_data.shape
        LOGGER.debug(
            f"Creating image plot for {n_rows}x{n_cols} array ({n_rows * n_cols:,} pixels)"
        )

        # Calculate physical dimensions if pixel_size is available
        # Adjust pixel_size for downsampling factor
        # pixel_size is in mm, convert to µm for display
        if pixel_size:
            effective_pixel_size = pixel_size * downsample_factor
            pixel_size_um = effective_pixel_size * 1000  # mm to µm
            x_range = (0, n_cols * pixel_size_um)
            y_range = (0, n_rows * pixel_size_um)
            x_label = "X (µm)"
            y_label = "Y (µm)"
        else:
            x_range = (0, n_cols)
            y_range = (0, n_rows)
            x_label = "X (pixels)"
            y_label = "Y (pixels)"

        # Calculate color range, ignoring NaN values
        # Use cached percentiles if available, otherwise calculate
        if cached_percentiles is not None:
            vmin, vmax = cached_percentiles
            LOGGER.debug(f"Using cached percentiles: p1={vmin:.3f}, p99={vmax:.3f}")
        else:
            t_percentile_start = time.perf_counter()
            valid_data = image_data[~np.isnan(image_data)]
            if len(valid_data) > 0:
                # Use percentiles to avoid outliers dominating the color scale
                vmin = np.percentile(valid_data, 1)
                vmax = np.percentile(valid_data, 99)
            else:
                vmin, vmax = 0, 1
            t_percentile = time.perf_counter() - t_percentile_start
            LOGGER.debug(
                f"Percentile calculation took {t_percentile:.3f}s ({len(valid_data):,} valid pixels)"
            )

        # Create color mapper
        color_mapper = LinearColorMapper(
            palette="Viridis256", low=vmin, high=vmax, nan_color="white"
        )

        # Create figure
        t_bokeh_start = time.perf_counter()
        p = figure(
            title=title,
            x_axis_label=x_label,
            y_axis_label=y_label,
            x_range=x_range,
            y_range=y_range,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="scale_width",
            aspect_ratio=n_cols / n_rows,
            match_aspect=True,
        )

        # Add image
        # Note: bokeh.plotting.figure.image expects data as [image] where image is 2D array
        # dw and dh are the width and height in data coordinates
        p.image(
            image=[image_data],
            x=0,
            y=0,
            dw=x_range[1],
            dh=y_range[1],
            color_mapper=color_mapper,
        )

        # Add colorbar
        color_bar = ColorBar(
            color_mapper=color_mapper,
            title=colorbar_label,
            location=(0, 0),
        )
        p.add_layout(color_bar, "right")

        # Style
        p.toolbar.logo = "grey"
        p.grid.visible = False

        t_bokeh = time.perf_counter() - t_bokeh_start
        t_total = time.perf_counter() - t_start
        LOGGER.debug(f"Bokeh figure creation took {t_bokeh:.3f}s")
        LOGGER.debug(f"Total image plot creation took {t_total:.3f}s")

        return p

    @staticmethod
    def _create_histogram_plot(
        image_data: np.ndarray,
        title: str = "Height Distribution",
        x_label: str = "Height",
        n_bins: int = 100,
        downsample_image_size: int = 10000,
    ):
        """
        Create a histogram plot for z-values (height distribution).

        Args:
            image_data: 2D numpy array of height/intensity values
            title: Plot title
            x_label: Label for x-axis
            n_bins: Number of bins for the histogram

        Returns:
            Bokeh figure with histogram plot
        """
        t_start = time.perf_counter()

        # Downsample for histogram calculation if data is very large
        # Use larger max_dimension since histogram doesn't need as much downsampling
        original_size = image_data.size
        if original_size > 10_000_000:  # > 10M pixels
            t_downsample_start = time.perf_counter()
            original_shape = image_data.shape
            image_data, downsample_factor = downsample_image_block_average(
                image_data, max_dimension=downsample_image_size
            )
            t_downsample = time.perf_counter() - t_downsample_start
            LOGGER.debug(
                f"Downsampled histogram data from {original_shape[0]}x{original_shape[1]} to "
                f"{image_data.shape[0]}x{image_data.shape[1]} ({t_downsample:.3f}s)"
            )

        # Get valid (non-NaN) data
        valid_data = image_data[~np.isnan(image_data)]
        LOGGER.debug(f"Creating histogram for {len(valid_data):,} valid pixels with {n_bins} bins")

        if len(valid_data) == 0:
            # Create empty plot if no valid data
            LOGGER.warning("No valid data for histogram, creating empty plot")
            p = figure(
                title=title,
                x_axis_label=x_label,
                y_axis_label="Count",
                sizing_mode="scale_width",
                height=300,
            )
            return p

        # Compute histogram
        t_hist_start = time.perf_counter()
        hist, edges = np.histogram(valid_data, bins=n_bins)
        t_hist = time.perf_counter() - t_hist_start
        LOGGER.debug(f"Histogram computation took {t_hist:.3f}s")

        # Create figure
        t_bokeh_start = time.perf_counter()
        p = figure(
            title=title,
            x_axis_label=x_label,
            y_axis_label="Count",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            sizing_mode="scale_width",
            height=300,
        )

        # Plot histogram as quads
        p.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color="navy",
            line_color="white",
            alpha=0.7,
        )

        # Style
        p.toolbar.logo = "grey"

        t_bokeh = time.perf_counter() - t_bokeh_start
        t_total = time.perf_counter() - t_start
        LOGGER.debug(f"Bokeh histogram figure creation took {t_bokeh:.3f}s")
        LOGGER.debug(f"Total histogram creation took {t_total:.3f}s")

        return p

    def generate_profiling_plot(self, filepath=None):
        """Generate the profiling plot from the associated file.

        Args:
            filepath: Optional path to the file to plot. If not provided,
                     uses the file_id from self.data to look up the file.
        """
        t_start_total = time.perf_counter()

        # Get file path either from parameter or from database lookup
        if filepath is not None:
            file_path = Path(filepath)
        else:
            if "file_id" not in self.data:
                LOGGER.debug("No file_id in block data, skipping plot generation")
                return
            t_file_lookup_start = time.perf_counter()
            file_info = get_file_info_by_id(self.data["file_id"], update_if_live=True)
            file_path = Path(file_info["location"])
            t_file_lookup = time.perf_counter() - t_file_lookup_start
            LOGGER.debug(f"File lookup took {t_file_lookup:.3f}s")

        LOGGER.info(f"Generating profiling plot for: {file_path.name}")

        ext = file_path.suffix.lower()

        if ext not in self.accepted_file_extensions:
            warnings.warn(
                f"Unsupported file extension (must be one of {self.accepted_file_extensions}, not {ext})"
            )
            return

        try:
            # Load the Wyko ASC file (with automatic caching)
            t_load_start = time.perf_counter()
            result = load_wyko_asc_cached(file_path, load_intensity=False, progress=False)
            t_load = time.perf_counter() - t_load_start
            LOGGER.info(f"File loading completed in {t_load:.3f}s")

            # Get the height data
            height_data = result["raw_data"]
            pixel_size = result["metadata"].get("pixel_size")
            data_size_mb = height_data.nbytes / (1024 * 1024)
            LOGGER.debug(f"Loaded data: {height_data.shape} array, {data_size_mb:.2f} MB")

            # Downsample for visualization if needed
            t_downsample_start = time.perf_counter()
            original_shape = height_data.shape
            height_data_downsampled, downsample_factor = downsample_image_block_average(
                height_data, max_dimension=self.downsample_image_size
            )
            t_downsample = time.perf_counter() - t_downsample_start

            if downsample_factor > 1.0:
                LOGGER.info(
                    f"Downsampled from {original_shape[0]}x{original_shape[1]} to "
                    f"{height_data_downsampled.shape[0]}x{height_data_downsampled.shape[1]} "
                    f"({t_downsample:.3f}s)"
                )

            # Apply baseline correction using median of downsampled data
            t_baseline_start = time.perf_counter()
            baseline = np.nanmedian(height_data_downsampled)
            height_data_downsampled = height_data_downsampled - baseline
            t_baseline = time.perf_counter() - t_baseline_start
            LOGGER.info(f"Baseline correction applied: baseline={baseline:.3f} ({t_baseline:.3f}s)")

            # Also apply baseline to original data for histogram
            height_data = height_data - baseline

            # Get cached percentiles if available
            cached_percentiles = None
            if "percentile_1" in result["metadata"] and "percentile_99" in result["metadata"]:
                cached_percentiles = (
                    result["metadata"]["percentile_1"],
                    result["metadata"]["percentile_99"],
                )

            # Create the 2D image plot
            t_image_start = time.perf_counter()
            image_plot = self._create_image_plot(
                height_data_downsampled,
                pixel_size=pixel_size,
                title="Surface Height Profile",
                colorbar_label="Height",
                cached_percentiles=cached_percentiles,
                downsample_factor=downsample_factor,
            )
            t_image = time.perf_counter() - t_image_start
            LOGGER.info(f"Image plot creation completed in {t_image:.3f}s")

            # Create the histogram plot
            t_hist_start = time.perf_counter()
            histogram_plot = self._create_histogram_plot(
                height_data,
                title="Height Distribution",
                x_label="Height (µm)",
                n_bins=100,
                downsample_image_size=self.downsample_image_size * 2,
            )
            t_hist = time.perf_counter() - t_hist_start
            LOGGER.info(f"Histogram plot creation completed in {t_hist:.3f}s")

            # Combine plots in a vertical layout
            t_layout_start = time.perf_counter()
            layout = column(image_plot, histogram_plot, sizing_mode="scale_width")
            t_layout = time.perf_counter() - t_layout_start
            LOGGER.debug(f"Layout creation took {t_layout:.3f}s")

            # Store as bokeh JSON
            t_json_start = time.perf_counter()
            self.data["bokeh_plot_data"] = bokeh.embed.json_item(layout, theme=DATALAB_BOKEH_THEME)
            t_json = time.perf_counter() - t_json_start
            json_size_kb = len(str(self.data["bokeh_plot_data"])) / 1024
            LOGGER.info(f"JSON serialization completed in {t_json:.3f}s ({json_size_kb:.1f} KB)")

            t_total = time.perf_counter() - t_start_total
            LOGGER.info(f"Total plot generation completed in {t_total:.3f}s")

        except Exception as e:
            LOGGER.error(f"Error loading profiling data: {e}", exc_info=True)
            warnings.warn(f"Error loading profiling data: {e}")
            return
