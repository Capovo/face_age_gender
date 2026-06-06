import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from tqdm import tqdm

from dataset import UTKFaceDataset, ensure_split_csvs, get_split_dirs
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def dataset_ages(dataset):
    ages = []
    for index in range(len(dataset)):
        ages.append(float(dataset[index]["age"].item()))
    return np.asarray(ages, dtype=np.float32)


def build_train_val_datasets(args):
    split_dirs = get_split_dirs(args.data_dir)
    generator = torch.Generator().manual_seed(args.seed)

    if split_dirs is not None:
        train_dataset = UTKFaceDataset(split_dirs["train"], labels_csv=args.labels_csv)
        if split_dirs["val"] is not None:
            val_dataset = UTKFaceDataset(split_dirs["val"], labels_csv=args.labels_csv)
            return train_dataset, val_dataset

        val_size = max(1, int(len(train_dataset) * 0.1))
        train_size = len(train_dataset) - val_size
        return random_split(train_dataset, [train_size, val_size], generator=generator)

    split_paths = ensure_split_csvs(
        args.data_dir,
        labels_csv=args.labels_csv,
        split_dir=args.split_dir,
        seed=args.seed,
        force=args.recreate_splits,
    )
    train_dataset = UTKFaceDataset(args.data_dir, labels_csv=args.labels_csv, split_csv=split_paths["train"])
    val_dataset = UTKFaceDataset(args.data_dir, labels_csv=args.labels_csv, split_csv=split_paths["val"])
    return train_dataset, val_dataset


def train_one_epoch(model, loader, optimizer, device, age_mean, age_std):
    model.train()
    age_loss_fn = nn.L1Loss()
    cls_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        ages = ((batch["age"] - age_mean) / age_std).to(device)
        genders = batch["gender"].to(device)

        pred_age, pred_gender = model(images)
        loss = age_loss_fn(pred_age, ages) + cls_loss_fn(pred_gender, genders)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device, age_mean, age_std):
    model.eval()
    age_loss_fn = nn.L1Loss()
    cls_loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    true_ages = []
    pred_ages = []
    true_genders = []
    pred_genders = []

    for batch in tqdm(loader, desc="valid", leave=False):
        images = batch["image"].to(device)
        ages = ((batch["age"] - age_mean) / age_std).to(device)
        genders = batch["gender"].to(device)

        pred_age, pred_gender = model(images)
        loss = age_loss_fn(pred_age, ages) + cls_loss_fn(pred_gender, genders)
        total_loss += loss.item() * images.size(0)

        pred_age_original = pred_age.cpu().numpy() * age_std + age_mean
        true_ages.extend(batch["age"].cpu().tolist())
        pred_ages.extend(pred_age_original.tolist())
        true_genders.extend(batch["gender"].cpu().tolist())
        pred_genders.extend(pred_gender.argmax(dim=1).cpu().tolist())

    return {
        "val_loss": total_loss / len(loader.dataset),
        "val_age_mae": mean_absolute_error(true_ages, pred_ages),
        "val_gender_accuracy": accuracy_score(true_genders, pred_genders),
        "val_gender_f1": f1_score(true_genders, pred_genders, average="binary", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/UTKFace")
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recreate-splits", action="store_true")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="checkpoints/best_model.pt")
    parser.add_argument("--csv-log", default="training_log.csv")
    parser.add_argument("--run-log", default="training_run.log")
    args = parser.parse_args()

    setup_logging(args.run_log)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = build_train_val_datasets(args)
    train_ages = dataset_ages(train_set)
    age_mean = float(train_ages.mean())
    age_std = float(train_ages.std())
    if age_std == 0:
        age_std = 1.0
    logging.info("Using device=%s age_mean=%.4f age_std=%.4f", device, age_mean, age_std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(
        args.model,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_log_path = Path(args.csv_log)
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_age_mae",
                "val_gender_accuracy",
                "val_gender_f1",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, age_mean, age_std)
            metrics = validate(model, val_loader, device, age_mean, age_std)
            row = {"epoch": epoch, "train_loss": train_loss, **metrics}
            writer.writerow(row)
            file.flush()
            logging.info(
                "Epoch %03d: train_loss=%.4f, val_loss=%.4f, val_age_mae=%.4f, "
                "val_gender_accuracy=%.4f, val_gender_f1=%.4f",
                epoch,
                train_loss,
                metrics["val_loss"],
                metrics["val_age_mae"],
                metrics["val_gender_accuracy"],
                metrics["val_gender_f1"],
            )

            if metrics["val_loss"] < best_loss:
                best_loss = metrics["val_loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_name": args.model,
                        "pretrained": not args.no_pretrained,
                        "freeze_backbone": args.freeze_backbone,
                        "tasks": ["age", "gender"],
                        "age_mean": age_mean,
                        "age_std": age_std,
                    },
                    output_path,
                )
                logging.info("Saved best model to %s", output_path)


if __name__ == "__main__":
    main()
