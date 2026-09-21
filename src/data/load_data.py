"""Memory-conscious loader for the EPA annual air-quality CSV."""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
import pandas as pd

DEFAULT_RAW_PATH = Path("data/raw/air_quality.csv")

# Keep identifiers as strings so leading zeroes are not lost.
ID_DTYPES = {
    "state_code": "string",
    "county_code": "string",
    "site_num": "string",
    "parameter_code": "Int64",
    "poc": "Int64",
    "year": "Int64",
}

def load_air_quality(
    path: str | Path = DEFAULT_RAW_PATH,
    *,
    chunksize: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load the EPA CSV. If chunksize is supplied, return a TextFileReader."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"EPA dataset not found: {path}. Put the full CSV at data/raw/air_quality.csv "
            "or pass a different path."
        )

    kwargs = dict(
        low_memory=False,
        dtype=ID_DTYPES,
        usecols=usecols,
    )
    if chunksize:
        return pd.read_csv(path, chunksize=chunksize, **kwargs)
    return pd.read_csv(path, **kwargs)

def iter_air_quality(
    path: str | Path = DEFAULT_RAW_PATH,
    *,
    chunksize: int = 250_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield chunks; preferred for the full ~300 MB file."""
    yield from load_air_quality(path, chunksize=chunksize, usecols=usecols)

if __name__ == "__main__":
    df = load_air_quality()
    print(f"Loaded shape: {df.shape}")
    print(df.head())
