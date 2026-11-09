from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = ROOT / "data"
GALLERY_DIR = DATA_DIR / "gallery"
OUTPUT_DIR = ROOT / "outputs"
UPLOAD_DIR = ROOT / "uploads"
for folder in (GALLERY_DIR, OUTPUT_DIR, UPLOAD_DIR):
    folder.mkdir(parents=True, exist_ok=True)
