"""Small CIFAR-10 loader for the dataset's official Python or binary archive."""

import hashlib
from pathlib import Path
import pickle
import tarfile
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import Dataset


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
ARCHIVE_MD5 = {
    "cifar-10-binary.tar.gz": "c32a1d4ab5d03f1284b67883e8d87530",
    "cifar-10-python.tar.gz": "c58f30108f718f92721af3b95e74349a",
}
DOWNLOAD_NAME = "cifar-10-binary.tar.gz"


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_archive(root: Path, download: bool) -> Path:
    for name, expected_md5 in ARCHIVE_MD5.items():
        archive = root / name
        if archive.exists():
            if _file_md5(archive) != expected_md5:
                raise ValueError(f"CIFAR-10 archive has an invalid checksum: {archive}")
            return archive
    if not download:
        raise FileNotFoundError(
            f"CIFAR-10 archive not found in {root}; run training to download it"
        )
    root.mkdir(parents=True, exist_ok=True)
    archive = root / DOWNLOAD_NAME
    partial = root / (DOWNLOAD_NAME + ".part")
    digest = hashlib.md5()
    try:
        with urlopen(CIFAR10_URL, timeout=60) as response, partial.open("wb") as destination:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                destination.write(block)
                digest.update(block)
        if digest.hexdigest() != ARCHIVE_MD5[DOWNLOAD_NAME]:
            raise ValueError("Downloaded CIFAR-10 archive failed its checksum")
        partial.replace(archive)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return archive


class CIFAR10Images(Dataset[torch.Tensor]):
    """Unlabelled CIFAR-10 RGB images as float tensors in [0, 1]."""

    def __init__(self, root: str | Path, train: bool, download: bool = False) -> None:
        archive = _ensure_archive(Path(root), download)
        is_python_archive = archive.name == "cifar-10-python.tar.gz"
        names = (
            [f"data_batch_{index}" + ("" if is_python_archive else ".bin") for index in range(1, 6)]
            if train
            else ["test_batch" + ("" if is_python_archive else ".bin")]
        )
        batches = []
        with tarfile.open(archive, "r:gz") as bundle:
            for name in names:
                prefix = "cifar-10-batches-py" if is_python_archive else "cifar-10-batches-bin"
                member = bundle.extractfile(f"{prefix}/{name}")
                if member is None:
                    raise ValueError(f"Missing CIFAR-10 file: {name}")
                if is_python_archive:
                    # The archive has been verified against the official MD5 above.
                    pixels = pickle.load(member, encoding="bytes")[b"data"]
                    if pixels.shape != (10_000, 3_072) or pixels.dtype != np.uint8:
                        raise ValueError(f"Unexpected CIFAR-10 batch shape: {name}")
                else:
                    raw = member.read()
                    if len(raw) != 10_000 * 3_073:
                        raise ValueError(f"Unexpected CIFAR-10 batch size: {name}")
                    rows = np.frombuffer(raw, dtype=np.uint8).reshape(10_000, 3_073)
                    pixels = rows[:, 1:]
                batches.append(pixels.reshape(10_000, 3, 32, 32))
        self.images = np.concatenate(batches, axis=0)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.images[index].copy()).float().div_(255.0)
