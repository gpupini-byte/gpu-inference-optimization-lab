"""Fetch the small public-domain validation set and strip EXIF metadata."""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "real_validation"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for record in manifest["images"]:
        target = DATA_DIR / record["file"]
        if not target.exists():
            request = Request(record["source_image"], headers={"User-Agent": "GPUInferenceOptimizationLab/1.0 (validation fixture fetch)"})
            with urlopen(request, timeout=30) as response:
                source_bytes = response.read()
            with Image.open(BytesIO(source_bytes)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.save(target, format="JPEG", quality=95, optimize=True)
        content = target.read_bytes()
        record["sha256"] = hashlib.sha256(content).hexdigest()
        with Image.open(BytesIO(content)) as image:
            record["image_size"] = list(image.size)
        print(f"{record['file']}: {record['sha256']}")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
