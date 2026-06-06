import argparse
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UTKFaceDataset, ensure_split_csvs, get_split_dirs
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_eval_dataset(args):
    split_dirs = get_split_dirs(args.data_dir)
    if split_dirs is not None:
        return UTKFaceDataset(split_dirs[args.split], labels_csv=args.labels_csv)

    split_paths = ensure_split_csvs(
        args.data_dir,
        labels_csv=args.labels_csv,
        split_dir=args.split_dir,
        seed=args.seed,
    )
    return UTKFaceDataset(args.data_dir, labels_csv=args.labels_csv, split_csv=split_paths[args.split])


@torch.no_grad()
def evaluate(model, loader, device, age_mean=0.0, age_std=1.0):
    model.eval()
    true_ages, pred_ages = [], []
    true_genders, pred_genders = [], []

    for batch in tqdm(loader, desc="evaluate"):
        images = batch["image"].to(device)
        age, gender = model(images)

        true_ages.extend(batch["age"].cpu().tolist())
        pred_ages.extend((age.cpu().numpy() * age_std + age_mean).tolist())
        true_genders.extend(batch["gender"].cpu().tolist())
        pred_genders.extend(gender.argmax(dim=1).cpu().tolist())

    return {
        "age_mae": mean_absolute_error(true_ages, pred_ages),
        "gender_accuracy": accuracy_score(true_genders, pred_genders),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/UTKFace")
    parser.add_argument("--labels-csv", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        checkpoint.get("model_name", "cnn"),
        pretrained=False,
        freeze_backbone=checkpoint.get("freeze_backbone", False),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    age_mean = checkpoint.get("age_mean", 0.0)
    age_std = checkpoint.get("age_std", 1.0)

    dataset = build_eval_dataset(args)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    metrics = evaluate(model, loader, device, age_mean=age_mean, age_std=age_std)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
