"""
Wyko ASCII Profilometer File Reader

Memory-efficient parser for Wyko .ASC profilometer data files containing
surface height (RAW_DATA) and intensity profiles.
"""

import time
from pathlib import Path
from typing import cast

import numpy as np
from pydatalab.logger import LOGGER


def parse_wyko_header(filepath: str | Path) -> dict[str, int | float | None]:
    """
    Parse Wyko ASC file header to extract metadata.

    Args:
        filepath: Path to the .ASC file

    Returns:
        Dictionary containing:
        - x_size: Number of X pixels
        - y_size: Number of Y pixels
        - pixel_size: Physical pixel size in mm
        - raw_data_start: Line number (1-indexed) where RAW_DATA begins
        - intensity_start: Line number (1-indexed) where Intensity begins (if present)

    Note: All line numbers are 1-indexed to match linecache.
    """
    t_start = time.perf_counter()
    filepath = Path(filepath)
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    LOGGER.debug(f"Parsing header for {filepath.name} ({file_size_mb:.1f} MB)")

    metadata: dict[str, int | float | None] = {
        "x_size": None,
        "y_size": None,
        "pixel_size": None,
        "raw_data_start": None,  # 1-indexed line number
        "intensity_start": None,  # 1-indexed line number
    }

    t_header_read_start = time.perf_counter()
    with open(filepath) as f:
        # Use 1-indexed line counting (start=1)
        for i, line in enumerate(f, start=1):
            line_stripped = line.strip()

            if line_stripped.startswith("X Size"):
                metadata["x_size"] = int(line_stripped.split()[-1])
            elif line_stripped.startswith("Y Size"):
                metadata["y_size"] = int(line_stripped.split()[-1])
            elif line_stripped.startswith("Pixel_size"):
                parts = line_stripped.split()
                metadata["pixel_size"] = float(parts[-1])
            elif line_stripped.startswith("RAW_DATA"):
                # If "RAW_DATA" header is on line i, data starts on line i+1
                metadata["raw_data_start"] = i + 1  # 1-indexed
            # Stop after reading header (first ~10 lines)
            # Don't scan the whole file
            if i > 10:
                break

    t_header_read = time.perf_counter() - t_header_read_start
    LOGGER.debug(f"Header read took {t_header_read:.3f}s")
    LOGGER.debug(
        f"Found dimensions: {metadata['x_size']}x{metadata['y_size']}, pixel_size: {metadata['pixel_size']} mm"
    )

    # Calculate Intensity start position from metadata
    # Intensity block header appears after all RAW_DATA rows
    # Structure: RAW_DATA rows + 1 line for "Intensity" header
    # Note: Intensity block definition appears mid-file, not in header,
    # so we calculate its position based on X Size
    if metadata["raw_data_start"] and metadata["x_size"]:
        raw_data_start = cast(int, metadata["raw_data_start"])  # 1-indexed
        x_size = cast(int, metadata["x_size"])

        # Intensity header line = raw_data_start + x_size rows (still 1-indexed)
        intensity_header_line = raw_data_start + x_size

        # Validate that "Intensity" actually appears at the calculated line
        # Open file and seek to the specific line efficiently (without reading entire file)
        t_intensity_check_start = time.perf_counter()
        with open(filepath) as f:
            # Skip to the intensity header line (convert 1-indexed to 0-indexed)
            for _ in range(intensity_header_line - 1):
                f.readline()

            # Read the intensity header line
            line = f.readline()

            if line.strip().startswith("Intensity"):
                # Data starts on the line after the "Intensity" header
                metadata["intensity_start"] = intensity_header_line + 1  # 1-indexed
                LOGGER.debug(f"Found Intensity block at line {intensity_header_line}")
            else:
                # Intensity block not found at expected location
                metadata["intensity_start"] = None
                LOGGER.debug("No Intensity block found")

        t_intensity_check = time.perf_counter() - t_intensity_check_start
        LOGGER.debug(f"Intensity check took {t_intensity_check:.3f}s")

    t_total = time.perf_counter() - t_start
    LOGGER.debug(f"Total header parsing took {t_total:.3f}s")

    return metadata


def load_wyko_profile_pandas_chunked(
    filepath: str | Path,
    start_line: int,
    n_rows: int,
    n_cols: int,
    name: str = "Profile",
    chunksize: int = 500,
) -> np.ndarray:
    """
    Load profile using pandas with chunking - balanced speed and memory.

    This is the recommended method for loading large Wyko files.

    Args:
        filepath: Path to the .ASC file
        start_line: Line number where data starts (1-indexed)
        n_rows: Number of rows to read
        n_cols: Number of columns expected per row
        name: Name for progress display
        chunksize: Number of rows per chunk

    Returns:
        numpy array of shape (n_rows, n_cols) with float32 dtype.
    """
    import pandas as pd

    # Pre-allocate output array
    data = np.empty((n_rows, n_cols), dtype=np.float32)

    # pandas.read_csv skiprows parameter is 0-indexed (number of lines to skip)
    # Convert 1-indexed line number to 0-indexed skip count
    chunks = pd.read_csv(
        filepath,
        skiprows=start_line - 1,  # If start_line=9, skip first 8 lines
        nrows=n_rows,
        sep="\t",
        header=None,
        na_values="Bad",
        dtype=np.float32,
        engine="c",
        chunksize=chunksize,
    )

    # Track where to write each chunk in the output array (0-indexed array position)
    row_offset = 0
    for chunk in chunks:
        chunk_data = chunk.values
        chunk_rows = chunk_data.shape[0]

        # Truncate columns if needed
        if chunk_data.shape[1] > n_cols:
            chunk_data = chunk_data[:, :n_cols]

        # Write chunk to output array at current offset position
        data[row_offset : row_offset + chunk_rows, :] = chunk_data
        row_offset += chunk_rows  # Move offset forward for next chunk

    return data


def load_wyko_asc(
    filepath: str | Path, load_intensity: bool = False, progress: bool = True
) -> dict:
    """
    Load a complete Wyko ASC profilometer file using pandas chunked reading.

    This gives a good tradeoff between time and memory efficiency.

    Args:
        filepath: Path to the .ASC file
        load_intensity: If True, also load the intensity profile
        progress: If True, show progress during loading

    Returns:
        Dictionary containing:
        - metadata: Header information
        - raw_data: Height profile as numpy array (n_rows x n_cols)
        - intensity: Intensity profile (only if load_intensity=True)

    Example:
        >>> result = load_wyko_asc('sample.ASC')
        >>> height = result['raw_data']
        >>> pixel_size = result['metadata']['pixel_size']

        >>> # Plot the data
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(height, cmap='viridis')
        >>> plt.colorbar(label='Height')
        >>> plt.show()
    """
    t_start_total = time.perf_counter()
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    LOGGER.info(f"Loading Wyko file: {filepath.name}")

    # Parse header
    t_metadata_start = time.perf_counter()
    metadata = parse_wyko_header(filepath)
    t_metadata = time.perf_counter() - t_metadata_start
    LOGGER.debug(f"Header parsing took {t_metadata:.3f}s")

    if metadata["x_size"] is None or metadata["y_size"] is None:
        raise ValueError("Could not parse X Size and Y Size from file header")

    if metadata["raw_data_start"] is None:
        raise ValueError("Could not find RAW_DATA block in file")

    # Note: Data layout in file is transposed
    # Rows in file = X Size, Columns in file = Y Size
    n_rows = cast(int, metadata["x_size"])
    n_cols = cast(int, metadata["y_size"])
    raw_data_start = cast(int, metadata["raw_data_start"])
    total_pixels = n_rows * n_cols

    LOGGER.info(f"File dimensions: {n_rows}x{n_cols} ({total_pixels:,} pixels)")

    if progress:
        print(f"Loading Wyko ASC file: {filepath.name}")
        print(f"  Dimensions: {n_rows} x {n_cols}")
        print(f"  Pixel size: {metadata['pixel_size']} mm")

    result = {"metadata": metadata}

    # Load RAW_DATA (height profile)
    t_load_raw_start = time.perf_counter()
    raw_data = load_wyko_profile_pandas_chunked(
        filepath,
        start_line=raw_data_start,
        n_rows=n_rows,
        n_cols=n_cols,
        name="RAW_DATA",
    )
    result["raw_data"] = raw_data
    t_load_raw = time.perf_counter() - t_load_raw_start
    data_size_mb = raw_data.nbytes / (1024 * 1024)
    load_speed_mb_s = data_size_mb / t_load_raw
    LOGGER.info(
        f"RAW_DATA loaded in {t_load_raw:.3f}s ({data_size_mb:.2f} MB at {load_speed_mb_s:.1f} MB/s)"
    )

    # Optionally load Intensity
    if load_intensity:
        if metadata["intensity_start"] is None:
            raise ValueError(
                "Intensity data requested but no Intensity block found in file. "
                "The file may not contain intensity data."
            )
        t_load_intensity_start = time.perf_counter()
        intensity_start = cast(int, metadata["intensity_start"])
        result["intensity"] = load_wyko_profile_pandas_chunked(
            filepath,
            start_line=intensity_start,
            n_rows=n_rows,
            n_cols=n_cols,
            name="Intensity",
        )
        t_load_intensity = time.perf_counter() - t_load_intensity_start
        LOGGER.info(f"Intensity loaded in {t_load_intensity:.3f}s")

    t_total = time.perf_counter() - t_start_total
    LOGGER.info(f"Total file loading completed in {t_total:.3f}s")

    return result


def save_wyko_cache(
    filepath: str | Path, result: dict, cache_path: str | Path | None = None
) -> Path:
    """
    Save loaded Wyko data as compressed numpy file for faster reloading.

    This function also pre-calculates and caches percentile values (1st and 99th)
    for the raw_data to speed up plotting operations.

    Args:
        filepath: Original ASC file path (used to generate cache name)
        result: Result dictionary from load_wyko_asc()
        cache_path: Optional explicit path for cache file

    Returns:
        Path to the saved cache file
    """
    if cache_path is None:
        cache_path = Path(filepath).with_suffix(".npz")
    else:
        cache_path = Path(cache_path)

    save_dict = {
        "raw_data": result["raw_data"],
        "x_size": result["metadata"]["x_size"],
        "y_size": result["metadata"]["y_size"],
        "pixel_size": result["metadata"]["pixel_size"],
    }

    # Calculate and cache percentiles for faster plotting
    t_percentile_start = time.perf_counter()
    raw_data = result["raw_data"]
    valid_data = raw_data[~np.isnan(raw_data)]

    if len(valid_data) > 0:
        percentile_1 = np.percentile(valid_data, 1)
        percentile_99 = np.percentile(valid_data, 99)
        save_dict["percentile_1"] = np.float32(percentile_1)
        save_dict["percentile_99"] = np.float32(percentile_99)
        t_percentile = time.perf_counter() - t_percentile_start
        LOGGER.debug(
            f"Calculated percentiles for caching in {t_percentile:.3f}s "
            f"(p1={percentile_1:.3f}, p99={percentile_99:.3f})"
        )
    else:
        # No valid data, store default values
        save_dict["percentile_1"] = np.float32(0.0)
        save_dict["percentile_99"] = np.float32(1.0)
        t_percentile = time.perf_counter() - t_percentile_start
        LOGGER.debug(f"No valid data for percentiles, using defaults ({t_percentile:.3f}s)")

    if "intensity" in result:
        save_dict["intensity"] = result["intensity"]

    np.savez_compressed(cache_path, **save_dict)
    print(f"Saved cache to: {cache_path}")

    return cache_path


def load_wyko_cache(cache_path: str | Path) -> dict:
    """
    Load Wyko data from a cached numpy file.

    Args:
        cache_path: Path to the .npz cache file

    Returns:
        Dictionary with same structure as load_wyko_asc(), plus optional
        cached percentile values in metadata if available.
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    cached = np.load(cache_path)

    result = {
        "metadata": {
            "x_size": int(cached["x_size"]),
            "y_size": int(cached["y_size"]),
            "pixel_size": float(cached["pixel_size"]),
        },
        "raw_data": cached["raw_data"],
    }

    if "intensity" in cached:
        result["intensity"] = cached["intensity"]

    return result


def load_wyko_asc_cached(
    filepath: str | Path, load_intensity: bool = False, progress: bool = False, reload: bool = False
) -> dict:
    """
    Load a Wyko ASC file with automatic caching for faster subsequent loads.

    This function checks for a cached .npz file alongside the original .ASC file.
    If the cache exists and is up-to-date, it loads from cache (much faster).
    Otherwise, it loads from the .ASC file and creates/updates the cache.

    Args:
        filepath: Path to the .ASC file
        load_intensity: If True, also load the intensity profile
        progress: If True, show progress during loading
        reload: If True, force reload from .ASC file and regenerate cache

    Returns:
        Dictionary containing:
        - metadata: Header information
        - raw_data: Height profile as numpy array (n_rows x n_cols)
        - intensity: Intensity profile (only if load_intensity=True)

    Example:
        >>> # First load: reads .ASC and creates cache (~30-90s for 3.5GB file)
        >>> result = load_wyko_asc_cached('sample.ASC')
        >>> # Subsequent loads: reads from cache (~2-5s)
        >>> result = load_wyko_asc_cached('sample.ASC')
    """
    t_start = time.perf_counter()
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Determine cache path
    cache_path = filepath.with_suffix(".npz")

    # Check if we should use the cache
    use_cache = False
    if not reload and cache_path.exists():
        # Check if cache is newer than source file
        asc_mtime = filepath.stat().st_mtime
        cache_mtime = cache_path.stat().st_mtime

        if cache_mtime >= asc_mtime:
            use_cache = True
            LOGGER.info(f"Loading from cache: {cache_path.name}")
        else:
            LOGGER.info("Cache is stale (older than source file), regenerating cache")

    # Load from cache if valid
    if use_cache:
        try:
            t_load_start = time.perf_counter()
            result = load_wyko_cache(cache_path)
            t_load = time.perf_counter() - t_load_start

            # Check if we need intensity data but cache doesn't have it
            if load_intensity and "intensity" not in result:
                LOGGER.warning(
                    "Cache exists but doesn't contain intensity data, reloading from .ASC"
                )
                use_cache = False
            else:
                data_size_mb = result["raw_data"].nbytes / (1024 * 1024)
                cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
                t_total = time.perf_counter() - t_start
                LOGGER.info(
                    f"Loaded from cache in {t_load:.3f}s (total: {t_total:.3f}s, "
                    f"{data_size_mb:.1f} MB data, {cache_size_mb:.1f} MB cache file)"
                )
                return result
        except Exception as e:
            LOGGER.warning(f"Failed to load cache ({e}), falling back to .ASC file")
            use_cache = False

    # Load from .ASC file (either cache wasn't valid or was disabled)
    LOGGER.info(f"Loading from .ASC file: {filepath.name}")
    result = load_wyko_asc(filepath, load_intensity=load_intensity, progress=progress)

    # Save cache for future use
    try:
        t_cache_start = time.perf_counter()
        cache_path = save_wyko_cache(filepath, result, cache_path=cache_path)
        t_cache = time.perf_counter() - t_cache_start
        cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
        LOGGER.info(f"Cache saved in {t_cache:.3f}s ({cache_size_mb:.1f} MB)")
    except Exception as e:
        LOGGER.warning(f"Failed to save cache: {e}")

    t_total = time.perf_counter() - t_start
    LOGGER.info(f"Total load time (with caching): {t_total:.3f}s")

    return result


# ============================================================================
# Legacy loading functions
# ============================================================================
# These are kept for reference and benchmarking, but are not actively used.
# The pandas chunked reader above provides the best speed/memory tradeoff.
# When new data files are supplied we can test if these are better suited for certain file sizes and update accordingly.


def load_wyko_profile(
    filepath: str | Path,
    start_line: int,
    n_rows: int,
    n_cols: int,
    name: str = "Profile",
    progress_interval: int = 500,
) -> np.ndarray:
    """
    LEGACY: Load a single profile from a Wyko ASC file using pure Python.

    Memory-efficient but slow. Use load_wyko_profile_pandas_chunked() instead.

    Args:
        filepath: Path to the .ASC file
        start_line: Line number where data starts (1-indexed)
        n_rows: Number of rows to read
        n_cols: Number of columns expected per row
        name: Name for progress display
        progress_interval: Print progress every N rows

    Returns:
        numpy array of shape (n_rows, n_cols) with float32 dtype.
    """
    # Pre-allocate with float32 to save memory (~50% reduction)
    data = np.empty((n_rows, n_cols), dtype=np.float32)

    with open(filepath) as f:
        # Skip to start_line (convert 1-indexed line number to 0-indexed skip count)
        # If start_line=9, we skip 8 lines to position at line 9
        for _ in range(start_line - 1):
            f.readline()

        # Read data rows
        for row_idx in range(n_rows):
            line = f.readline()
            if not line:
                print(f"Warning: Unexpected end of file at row {row_idx}")
                data[row_idx:, :] = np.nan
                break

            values = line.split()

            # Process values, handling 'Bad' markers
            col_idx = 0
            for val in values:
                if col_idx >= n_cols:
                    break
                if val == "Bad":
                    data[row_idx, col_idx] = np.nan
                else:
                    try:
                        data[row_idx, col_idx] = float(val)
                    except ValueError:
                        data[row_idx, col_idx] = np.nan
                col_idx += 1

            # Fill remaining columns with NaN if row was short
            if col_idx < n_cols:
                data[row_idx, col_idx:] = np.nan

            # Progress indicator
            if progress_interval and row_idx % progress_interval == 0:
                print(f"{name}: {row_idx}/{n_rows} rows loaded", end="\r")

    if progress_interval:
        print(f"\n{name} loaded: shape={data.shape}, dtype={data.dtype}")

    return data


def load_wyko_profile_pandas(
    filepath: str | Path,
    start_line: int,
    n_rows: int,
    n_cols: int,
    name: str = "Profile",
) -> np.ndarray:
    """
    LEGACY: Load profile using pandas without chunking.

    Fastest but uses more memory. Use load_wyko_profile_pandas_chunked() instead
    for better memory efficiency.

    Args:
        filepath: Path to the .ASC file
        start_line: Line number where data starts (1-indexed)
        n_rows: Number of rows to read
        n_cols: Number of columns expected per row
        name: Name for progress display

    Returns:
        numpy array of shape (n_rows, n_cols) with float32 dtype.
    """
    import pandas as pd

    print(f"{name}: Loading with pandas...")

    # pandas.read_csv skiprows parameter is 0-indexed (number of lines to skip)
    # Convert 1-indexed line number to 0-indexed skip count
    df = pd.read_csv(
        filepath,
        skiprows=start_line - 1,  # If start_line=9, skip first 8 lines
        nrows=n_rows,
        sep="\t",
        header=None,
        na_values="Bad",
        dtype=np.float32,
        engine="c",  # Use C parser for speed
    )

    data = df.values

    # Ensure correct shape (truncate extra columns if needed)
    if data.shape[1] > n_cols:
        data = data[:, :n_cols]

    print(f"{name} loaded: shape={data.shape}, dtype={data.dtype}")
    return data
