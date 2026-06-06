import csv
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


RACE_NAMES = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}

GENDER_NAMES = {
    0: "Male",
    1: "Female",
}


def parse_utkface_name(path):
    """Parse UTKFace filename: age_gender_race_date.jpg."""
    parts = Path(path).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid UTKFace filename: {path}")

    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])
    return age, gender, race


def find_image_path(root_dir, file_name):
    candidates = [
        root_dir / file_name,
        root_dir / Path(file_name).name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def resolve_labels_path(root_dir, labels_csv=None):
    root_dir = Path(root_dir)
    if labels_csv:
        return Path(labels_csv)

    candidates = [
        root_dir.parent / "utkface_labels.csv",
        root_dir.parent.parent / "utkface_labels.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def read_label_rows(root_dir, labels_csv=None):
    root_dir = Path(root_dir)
    labels_path = resolve_labels_path(root_dir, labels_csv)
    rows = []

    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                image_path = find_image_path(root_dir, row["file_name"])
                if image_path is None:
                    continue
                rows.append(
                    {
                        "file_name": image_path.name,
                        "age": int(row["age"]),
                        "gender": int(row["gender"]),
                        "race": int(row["race"]),
                        "source_name": row.get("source_name", ""),
                    }
                )
        return rows

    for image_path in sorted(
        list(root_dir.glob("*.jpg")) + list(root_dir.glob("*.jpeg")) + list(root_dir.glob("*.png"))
    ):
        try:
            age, gender, race = parse_utkface_name(image_path)
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "file_name": image_path.name,
                "age": age,
                "gender": gender,
                "race": race,
                "source_name": image_path.name,
            }
        )
    return rows


def write_split_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["file_name", "age", "gender", "race", "source_name"])
        writer.writeheader()
        writer.writerows(rows)


def ensure_split_csvs(root_dir, labels_csv=None, split_dir=None, seed=42, force=False):
    root_dir = Path(root_dir)
    split_dir = Path(split_dir) if split_dir else root_dir.parent / "splits"
    split_paths = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
    }

    if not force and all(path.exists() for path in split_paths.values()):
        return split_paths

    rows = read_label_rows(root_dir, labels_csv)
    if len(rows) == 0:
        raise RuntimeError(f"No labeled images found in {root_dir}")

    indices = list(range(len(rows)))
    genders = [row["gender"] for row in rows]
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.1,
        random_state=seed,
        stratify=genders,
    )
    train_val_genders = [genders[index] for index in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=1 / 9,
        random_state=seed,
        stratify=train_val_genders,
    )

    write_split_csv(split_paths["train"], [rows[index] for index in train_idx])
    write_split_csv(split_paths["val"], [rows[index] for index in val_idx])
    write_split_csv(split_paths["test"], [rows[index] for index in test_idx])
    return split_paths


def get_split_dirs(root_dir):
    root_dir = Path(root_dir)
    train_dir = root_dir / "train"
    val_dir = root_dir / "val"
    if not val_dir.exists():
        val_dir = root_dir / "valid"
    test_dir = root_dir / "test"

    if train_dir.exists() and test_dir.exists():
        return {
            "train": train_dir,
            "val": val_dir if val_dir.exists() else None,
            "test": test_dir,
        }
    return None


class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, labels_csv=None, split_csv=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or self.default_transform()
        self.labels = None

        labels_path = Path(split_csv) if split_csv else resolve_labels_path(self.root_dir, labels_csv)
        if labels_path.exists():
            self.labels = {}
            image_paths = []
            with labels_path.open("r", encoding="utf-8", newline="") as file:
                for row in csv.DictReader(file):
                    image_path = find_image_path(self.root_dir, row["file_name"])
                    if image_path is None:
                        continue

                    self.labels[image_path.name] = {
                        "age": int(row["age"]),
                        "gender": int(row["gender"]),
                        "race": int(row["race"]),
                    }
                    image_paths.append(image_path)

            self.image_paths = sorted(image_paths)
            if len(self.image_paths) == 0:
                raise RuntimeError(
                    f"Found labels file {labels_path}, but no matching images in {self.root_dir}"
                )
            return

        self.image_paths = sorted(
            list(self.root_dir.glob("*.jpg"))
            + list(self.root_dir.glob("*.jpeg"))
            + list(self.root_dir.glob("*.png"))
        )
        valid_paths = []
        for path in self.image_paths:
            try:
                parse_utkface_name(path)
                valid_paths.append(path)
            except (ValueError, IndexError):
                continue
        self.image_paths = valid_paths
        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No valid labeled images found in {self.root_dir}. "
                f"For the Hugging Face download, keep {self.root_dir.parent / 'utkface_labels.csv'} "
                "or pass --labels-csv explicitly."
            )

    @staticmethod
    def default_transform():
        return transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        if self.labels is None:
            age, gender, race = parse_utkface_name(image_path)
        else:
            label = self.labels[image_path.name]
            age = label["age"]
            gender = label["gender"]
            race = label["race"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "age": torch.tensor(age, dtype=torch.float32),
            "gender": torch.tensor(gender, dtype=torch.long),
            "race": torch.tensor(race, dtype=torch.long),
            "path": str(image_path),
        }
