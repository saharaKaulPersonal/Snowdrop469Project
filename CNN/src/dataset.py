from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class MultiLabelSample:
    image_path: Path
    labels: np.ndarray


class MultiLabelImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        class_names: Sequence[str],
        transform=None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.class_names = list(class_names)
        self.transform = transform
        self.samples = self._read_samples()

    def _read_samples(self) -> List[MultiLabelSample]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8") as file:
            header = file.readline().strip().split(",")
            if len(header) < 2:
                raise ValueError(
                    "CSV requires at least image_path and one class column."
                )

            expected = ["image_path", *self.class_names]
            if header != expected:
                raise ValueError(
                    f"CSV header mismatch. Expected {expected}, got {header}"
                )

            samples: List[MultiLabelSample] = []
            for line_number, line in enumerate(file, start=2):
                line = line.strip()
                if not line:
                    continue
                fields = line.split(",")
                if len(fields) != len(header):
                    raise ValueError(
                        f"Malformed CSV row at line {line_number}: {line}"
                    )

                rel_image = fields[0]
                label_values = np.asarray(fields[1:], dtype=np.float32)
                if np.any((label_values != 0) & (label_values != 1)):
                    raise ValueError(
                        f"Labels must be binary 0/1 at line {line_number}."
                    )

                image_path = self.image_root / rel_image
                samples.append(MultiLabelSample(image_path=image_path, labels=label_values))

        if not samples:
            raise ValueError(f"No samples loaded from {self.csv_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if not sample.image_path.exists():
            raise FileNotFoundError(f"Image not found: {sample.image_path}")

        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        labels = torch.from_numpy(sample.labels)
        return image, labels
