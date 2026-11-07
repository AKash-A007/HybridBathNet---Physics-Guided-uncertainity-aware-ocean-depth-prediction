"""Verify extracted Sentinel-2 files against their original SAFE manifests."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    reports = []
    for folder in sorted(Path('data/raw/sentinel2').glob('*.SAFE')):
        checked, failures = 0, []
        for obj in ET.parse(folder / 'manifest.safe').getroot().findall('.//dataObject'):
            stream = obj.find('byteStream')
            location = stream.find('fileLocation')
            checksum = stream.find('checksum')
            path = folder / location.attrib['href']
            if not path.is_file():
                failures.append({'file': str(path), 'reason': 'missing'})
                continue
            if path.stat().st_size != int(stream.attrib['size']):
                failures.append({'file': str(path), 'reason': 'size mismatch'})
                continue
            algorithm = checksum.attrib['checksumName'].lower().replace('-', '_')
            with path.open('rb') as source:
                digest = hashlib.file_digest(source, algorithm).hexdigest()
            if digest.lower() != checksum.text.lower():
                failures.append({'file': str(path), 'reason': 'checksum mismatch'})
            checked += 1
        reports.append({'product': folder.name, 'files_checked': checked, 'failures': failures})
        print(f'{folder.name}: {checked} checked, {len(failures)} failures', flush=True)
    Path('runs/2026-11-04/manifest_verification.json').write_text(json.dumps(reports, indent=2))
    if any(row['failures'] for row in reports):
        raise SystemExit('Manifest verification failed')


if __name__ == '__main__':
    main()
