from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "ai-model" / "configs" / "data.yaml"
DATASET = REPO_ROOT / "yolo_dataset"
OUTPUT = REPO_ROOT / "ai-model" / "configs" / "data.local.yaml"

if not DATASET.is_dir():
    raise SystemExit(
        f"Dataset not found: {DATASET}\n"
        "Place or link the YOLO dataset at the repository root."
    )

if "__DATASET_ROOT__" not in TEMPLATE.read_text():
    raise SystemExit("Dataset template is missing its path placeholder.")

content = TEMPLATE.read_text().replace(
    "__DATASET_ROOT__",
    str(DATASET.resolve())
)

OUTPUT.write_text(content)
print(f"Created: {OUTPUT}")
print(f"Dataset: {DATASET.resolve()}")
