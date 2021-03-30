from pathlib import Path
from collections import namedtuple


main = Path.cwd()
assert main.name == "project", main.name

data = main / "data"
assert data.exists(), f"Data folder not found. ({data})"

models = main / "Models"
assert models.exists(), f"Models folder not found. ({models})"

self_training_parents = models / "SelfTrainingParents"
assert models.exists(), (
    f"Self Training Parents folder not found. ({self_training_parents})")

figures = main / "figures"
assert figures.exists(), f"Figures folder not found. ({figures})"

progress_report = figures / "progress_reports" / "current"
assert progress_report.exists(), (
    f"Progress Report folder not found. ({progress_report})")


ImageTypeTuple = namedtuple(
    "ImageTypeTuple",
    ("images", "cell", "modified_cell", "mitochondria", "nuclei", "nucleoli")
)


_images = data / "images"
assert _images.exists(), f"Image folder not found. ({_images})"
images = ImageTypeTuple(
    _images / "images",
    _images / "cell",
    _images / "modified_cell",
    _images / "mitochondria",
    _images / "nuclei",
    _images / "nucleoli",
)


_raw = Path("/home/spencer/ohsu/research/em/data/")
assert _raw.exists(), f"Original data folder not found. ({raw})"
raw = ImageTypeTuple(
    _raw / "Images",
    _raw / "together1",
    None,  # modified_cell
    _raw / "Mitochondria",
    _raw / "Nuclei",
    _raw / "Nucleoli",
)

