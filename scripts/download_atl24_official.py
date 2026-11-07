"""Download AOI-intersecting NSIDC ATL24 V002 granules using a local token.

The token is read from the current Windows user's profile and never printed or
saved to the repository. A temporary plain token file is removed on success.
"""
from pathlib import Path
import json
import os
import subprocess

import requests

OUT = Path("runs/2026-11-05/nearshore_error_reference_audit_v1")
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def load_token():
    plain = Path.home() / ".earthdata_token"
    if plain.exists():
        token = plain.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError("The Earthdata token file is empty")
        return token
    command = ("$ErrorActionPreference='Stop'; "
               "$s=Get-Content -LiteralPath (Join-Path $env:USERPROFILE "
               "'.earthdata_token.dpapi') -Raw | ConvertTo-SecureString; "
               "[System.Net.NetworkCredential]::new('', $s).Password")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                            capture_output=True, text=True, check=True)
    token = result.stdout.strip()
    if not token:
        raise RuntimeError("The Earthdata token file is empty")
    return token


def main():
    if os.name != "nt":
        raise RuntimeError("This script expects Windows DPAPI under the same user account")
    token_file = Path.home() / ".earthdata_token.dpapi"
    plain_file = Path.home() / ".earthdata_token"
    if not token_file.exists() and not plain_file.exists():
        raise FileNotFoundError("Save an Earthdata token under the Windows user profile first")
    token = load_token()
    candidates = json.loads((OUT / "atl24_cmr_candidates.json").read_text(encoding="utf-8-sig"))
    dest = OUT / "atl24_v002_official"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    session = requests.Session()
    for item in candidates:
        name = item["granule"]
        path = dest / name
        if path.exists():
            with path.open("rb") as handle:
                if handle.read(8) != HDF5_MAGIC:
                    raise ValueError(f"Existing file is not HDF5: {path}")
            rows.append({"granule": name, "status": "existing_valid_hdf5", "bytes": path.stat().st_size})
            continue
        part = dest / (name + ".part")
        try:
            with session.get(item["url"], headers={"Authorization": f"Bearer {token}"},
                             stream=True, timeout=(20, 120)) as response:
                response.raise_for_status()
                chunks = response.iter_content(chunk_size=1024 * 1024)
                first = next(chunks, b"")
                if not first.startswith(HDF5_MAGIC):
                    raise ValueError(f"{name}: response is not HDF5; check Earthdata authorization")
                with part.open("wb") as handle:
                    handle.write(first)
                    for chunk in chunks:
                        if chunk:
                            handle.write(chunk)
            part.replace(path)
            rows.append({"granule": name, "status": "downloaded_valid_hdf5", "bytes": path.stat().st_size})
            print(f"Downloaded {name}: {path.stat().st_size} bytes", flush=True)
        except Exception as error:
            rows.append({"granule": name, "status": "failed", "reason": str(error)})
            print(f"Download failed for {name}: {type(error).__name__}", flush=True)
            break
    (OUT / "atl24_v002_download_status.json").write_text(json.dumps({
        "aoi_bbox_lonlat": [76.30, 8.90, 76.55, 9.20],
        "catalogue_intersection_only": "Whole granules must be spatially subset after download",
        "granules": rows}, indent=2), encoding="utf-8")
    if any(row["status"] == "failed" for row in rows):
        raise RuntimeError("Official ATL24 download failed; see status without credentials")
    if plain_file.exists():
        plain_file.unlink()


if __name__ == "__main__":
    main()
