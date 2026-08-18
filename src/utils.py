"""Shared utilities: config loading, logging, small geo helpers."""
import logging
from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    """Return the EPSG code string for the UTM zone containing (lon, lat).
    Use this to set project.crs in config.yaml for your AOI."""
    zone = int((lon + 180) / 6) + 1
    hemisphere = 326 if lat >= 0 else 327  # 326xx = North, 327xx = South
    return f"EPSG:{hemisphere}{zone:02d}"
