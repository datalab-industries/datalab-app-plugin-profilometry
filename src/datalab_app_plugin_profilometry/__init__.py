"""
Profiling apps for surface profilometry data analysis.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("datalab-app-plugin-profilometry")
except PackageNotFoundError:
    __version__ = "develop"

del _version, PackageNotFoundError

from .blocks import ProfilingBlock
from .wyko_reader import (
    load_wyko_asc,
    load_wyko_asc_cached,
    load_wyko_cache,
    load_wyko_profile,
    load_wyko_profile_pandas,
    load_wyko_profile_pandas_chunked,
    parse_wyko_header,
    save_wyko_cache,
)

__all__ = [
    "ProfilingBlock",
    "load_wyko_asc",
    "load_wyko_asc_cached",
    "load_wyko_cache",
    "load_wyko_profile",
    "load_wyko_profile_pandas",
    "load_wyko_profile_pandas_chunked",
    "parse_wyko_header",
    "save_wyko_cache",
]
