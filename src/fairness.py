import argparse
from collections import defaultdict
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error
from torch.utils.data import DataLoader

from dataset import RACE_NAMES, UTKFaceDataset, ensure_split_csvs, get_split_dirs
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
def fairness_by_race(model, loader, device, age_mean=0.0, age_std=1.0):
    model.eval()
    groups = defaultdict(lambda: {"true_age": [], "pred_age": [], "true_gender": [], "pred_gender": []})

    for batch in loader:
        images = batch["image"].to(device)
        pred_age, pred_gender = model(images)
        pred_gender = pred_gender.argmax(dim=1).cpu().tolist()

        for i, race_id in enumerate(batch["race"].tolist()):
            group = groups[race_id]
            group["true_age"].append(batch["age"][i].item())
            group["pred_age"].append(pred_age[i].cpu().item() * age_std + age_mean)
            group["true_gender"].append(batch["gender"][i].item())
            group["pred_gender"].append(pred_gender[i])

    report = {}
    for race_id, values in groups.items():
        report[RACE_NAMES.get(race_id, str(race_id))] = {
            "count": len(values["true_age"]),
            "age_mae": mean_absolute_error(values["true_age"], values["pred_age"]),
            "gender_accuracy": accuracy_score(values["true_gender"], values["pred_gender"]),
        }
    return report


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
    report = fairness_by_race(model, loader, device, age_mean=age_mean, age_std=age_std)

    for race_name, metrics in report.items():
        print(
            f"{race_name}: count={metrics['count']}, "
            f"age_mae={metrics['age_mae']:.4f}, "
            f"gender_accuracy={metrics['gender_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
