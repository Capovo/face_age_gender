import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from tqdm import tqdm

from dataset import ensure_split_csvs, get_split_dirs


HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def read_labels(data_dir, labels_csv):
    data_dir = Path(data_dir)
    labels_csv = Path(labels_csv) if labels_csv else data_dir.parent / "utkface_labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    samples = []
    with labels_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            image_path = data_dir / Path(row["file_name"]).name
            if not image_path.exists():
                continue
            samples.append(
                {
                    "path": image_path,
                    "age": float(row["age"]),
                    "gender": int(row["gender"]),
                }
            )

    if len(samples) == 0:
        raise RuntimeError(f"No labeled images found in {data_dir}")
    return samples


def build_split_samples(args, split):
    split_dirs = get_split_dirs(args.data_dir)
    if split_dirs is not None:
        split_dir = split_dirs[split]
        if split_dir is None:
            raise RuntimeError("Existing dataset has train/test directories but no val or valid directory.")
        return read_labels(split_dir, args.labels_csv)

    split_paths = ensure_split_csvs(
        args.data_dir,
        labels_csv=args.labels_csv,
        split_dir=args.split_dir,
        seed=args.seed,
        force=getattr(args, "recreate_splits", False),
    )
    return read_labels(args.data_dir, split_paths[split])


def extract_hog_feature(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = resize(image, image_size, anti_aliasing=True)
    gray = rgb2gray(image)
    return hog(gray, feature_vector=True, **HOG_PARAMS)


def build_features(samples, image_size):
    features = []
    ages = []
    genders = []
    paths = []
    for sample in tqdm(samples, desc="extract HOG"):
        features.append(extract_hog_feature(sample["path"], image_size))
        ages.append(sample["age"])
        genders.append(sample["gender"])
        paths.append(str(sample["path"]))

    return np.asarray(features), np.asarray(ages), np.asarray(genders), paths


def train(args):
    train_samples = build_split_samples(args, "train")
    val_samples = build_split_samples(args, "val")
    if args.limit is not None:
        train_samples = train_samples[: args.limit]
        val_samples = val_samples[: max(1, int(args.limit * 0.125))]

    train_x, train_ages, train_genders, _ = build_features(train_samples, (args.image_size, args.image_size))
    val_x, val_ages, val_genders, val_paths = build_features(val_samples, (args.image_size, args.image_size))

    gender_model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=args.gender_c, class_weight="balanced", max_iter=args.max_iter),
    )
    age_model = make_pipeline(
        StandardScaler(),
        LinearSVR(C=args.age_c, epsilon=args.age_epsilon, max_iter=args.max_iter),
    )

    gender_model.fit(train_x, train_genders)
    age_model.fit(train_x, train_ages)

    gender_pred = gender_model.predict(val_x)
    age_pred = age_model.predict(val_x)
    metrics = {
        "val_gender_accuracy": accuracy_score(val_genders, gender_pred),
        "val_age_mae": mean_absolute_error(val_ages, age_pred),
        "train_size": int(len(train_samples)),
        "val_size": int(len(val_samples)),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "gender_model": gender_model,
            "age_model": age_model,
            "hog_params": HOG_PARAMS,
            "image_size": args.image_size,
            "metrics": metrics,
            "val_paths": val_paths,
            "seed": args.seed,
        },
        output_path,
    )

    print(f"Saved HOG baseline to {output_path}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def evaluate(args):
    samples = build_split_samples(args, args.split)
    if args.limit is not None:
        samples = samples[: args.limit]

    bundle = joblib.load(args.checkpoint)
    image_size = bundle["image_size"]
    x, ages, genders, _ = build_features(samples, (image_size, image_size))

    gender_pred = bundle["gender_model"].predict(x)
    age_pred = bundle["age_model"].predict(x)
    metrics = {
        "gender_accuracy": accuracy_score(genders, gender_pred),
        "age_mae": mean_absolute_error(ages, age_pred),
        "samples": len(samples),
        "split": args.split,
    }

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="HOG + SVM/SVR baseline for UTKFace age and gender.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", default="data/UTKFace")
    train_parser.add_argument("--labels-csv", default=None)
    train_parser.add_argument("--split-dir", default=None)
    train_parser.add_argument("--recreate-splits", action="store_true")
    train_parser.add_argument("--output", default="checkpoints/hog_baseline.joblib")
    train_parser.add_argument("--limit", type=int, default=None)
    train_parser.add_argument("--image-size", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--gender-c", type=float, default=1.0)
    train_parser.add_argument("--age-c", type=float, default=1.0)
    train_parser.add_argument("--age-epsilon", type=float, default=1.0)
    train_parser.add_argument("--max-iter", type=int, default=5000)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--data-dir", default="data/UTKFace")
    eval_parser.add_argument("--labels-csv", default=None)
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.add_argument("--split-dir", default=None)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument("--checkpoint", default="checkpoints/hog_baseline.joblib")
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
