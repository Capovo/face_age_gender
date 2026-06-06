import argparse
import copy
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import RACE_NAMES, UTKFaceDataset, ensure_split_csvs
from model import build_model


HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


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


def make_datasets(args):
    split_paths = ensure_split_csvs(
        args.data_dir,
        labels_csv=args.labels_csv,
        split_dir=args.split_dir,
        seed=args.seed,
        force=args.recreate_splits,
    )
    datasets = {
        split: UTKFaceDataset(args.data_dir, labels_csv=args.labels_csv, split_csv=path)
        for split, path in split_paths.items()
    }
    if args.limit is not None:
        datasets["train"] = Subset(datasets["train"], range(min(args.limit, len(datasets["train"]))))
        val_limit = max(1, min(len(datasets["val"]), args.limit // 8))
        test_limit = max(1, min(len(datasets["test"]), args.limit // 8))
        datasets["val"] = Subset(datasets["val"], range(val_limit))
        datasets["test"] = Subset(datasets["test"], range(test_limit))
    return datasets


def extract_hog_feature(image_path, image_size):
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = resize(image, (image_size, image_size), anti_aliasing=True)
    return hog(rgb2gray(image), feature_vector=True, **HOG_PARAMS).astype(np.float32)


def build_hog_arrays(dataset, image_size, limit=None):
    indices = range(len(dataset))
    features, ages, genders, races = [], [], [], []
    for index in tqdm(indices, desc="HOG"):
        item = dataset[index]
        features.append(extract_hog_feature(item["path"], image_size))
        ages.append(item["age"].item())
        genders.append(item["gender"].item())
        races.append(item["race"].item())
    return np.asarray(features), np.asarray(ages), np.asarray(genders), np.asarray(races)


def dataset_age_stats(dataset):
    ages = [float(dataset[index]["age"].item()) for index in range(len(dataset))]
    ages = np.asarray(ages, dtype=np.float32)
    age_mean = float(ages.mean())
    age_std = float(ages.std())
    return age_mean, age_std if age_std != 0 else 1.0


def metrics_dict(true_ages, pred_ages, true_genders, pred_genders):
    return {
        "Age MAE": mean_absolute_error(true_ages, pred_ages),
        "Gender Accuracy": accuracy_score(true_genders, pred_genders),
        "Gender F1-score": f1_score(true_genders, pred_genders, average="binary", zero_division=0),
    }


def train_hog_baseline(datasets, args):
    logging.info("Training HOG + SVM/SVR baseline")
    train_x, train_ages, train_genders, _ = build_hog_arrays(datasets["train"], args.hog_image_size, args.limit)
    val_x, val_ages, val_genders, _ = build_hog_arrays(datasets["val"], args.hog_image_size, args.limit)
    test_x, test_ages, test_genders, test_races = build_hog_arrays(datasets["test"], args.hog_image_size, args.limit)
    age_mean = float(train_ages.mean())
    age_std = float(train_ages.std())
    if age_std == 0:
        age_std = 1.0
    train_ages_norm = (train_ages - age_mean) / age_std

    gender_model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, class_weight="balanced", max_iter=args.hog_max_iter, random_state=args.seed),
    )
    age_model = make_pipeline(
        StandardScaler(),
        LinearSVR(C=1.0, epsilon=1.0, max_iter=args.hog_max_iter, random_state=args.seed),
    )
    gender_model.fit(train_x, train_genders)
    age_model.fit(train_x, train_ages_norm)

    val_pred_genders = gender_model.predict(val_x)
    val_pred_ages = age_model.predict(val_x) * age_std + age_mean
    pred_genders = gender_model.predict(test_x)
    pred_ages = age_model.predict(test_x) * age_std + age_mean
    val_metrics = metrics_dict(val_ages, val_pred_ages, val_genders, val_pred_genders)
    return {
        "method": "HOG + SVM/SVR",
        "metrics": metrics_dict(test_ages, pred_ages, test_genders, pred_genders),
        "fairness": fairness_from_arrays(test_ages, pred_ages, test_genders, pred_genders, test_races),
        "curve": {
            "Train Loss": [np.nan] * args.epochs,
            "Validation Age MAE": [val_metrics["Age MAE"]] * args.epochs,
            "Validation Gender Accuracy": [val_metrics["Gender Accuracy"]] * args.epochs,
        },
    }


def train_one_epoch(model, loader, optimizer, device, age_mean, age_std):
    model.train()
    age_loss_fn = nn.L1Loss()
    gender_loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        ages = ((batch["age"] - age_mean) / age_std).to(device)
        genders = batch["gender"].to(device)

        pred_ages, pred_genders = model(images)
        loss = age_loss_fn(pred_ages, ages) + gender_loss_fn(pred_genders, genders)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_torch_model(model, loader, device, age_mean, age_std):
    model.eval()
    true_ages, pred_ages, true_genders, pred_genders, races = [], [], [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        age_logits, gender_logits = model(images)
        true_ages.extend(batch["age"].cpu().tolist())
        pred_ages.extend((age_logits.cpu().numpy() * age_std + age_mean).tolist())
        true_genders.extend(batch["gender"].cpu().tolist())
        pred_genders.extend(gender_logits.argmax(dim=1).cpu().tolist())
        races.extend(batch["race"].cpu().tolist())

    return (
        np.asarray(true_ages),
        np.asarray(pred_ages),
        np.asarray(true_genders),
        np.asarray(pred_genders),
        np.asarray(races),
    )


def train_deep_model(method_name, model_name, datasets, args, pretrained=True, freeze_backbone=False):
    logging.info("Training %s", method_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    age_mean, age_std = dataset_age_stats(datasets["train"])
    logging.info("%s device=%s age_mean=%.4f age_std=%.4f", method_name, device, age_mean, age_std)

    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    curve = {"Train Loss": [], "Validation Age MAE": [], "Validation Gender Accuracy": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, age_mean, age_std)
        val_ages, val_pred_ages, val_genders, val_pred_genders, _ = evaluate_torch_model(
            model, val_loader, device, age_mean, age_std
        )
        val_metrics = metrics_dict(val_ages, val_pred_ages, val_genders, val_pred_genders)
        curve["Train Loss"].append(train_loss)
        curve["Validation Age MAE"].append(val_metrics["Age MAE"])
        curve["Validation Gender Accuracy"].append(val_metrics["Gender Accuracy"])

        score = val_metrics["Age MAE"] - 10 * val_metrics["Gender Accuracy"]
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

        logging.info(
            "%s epoch %d: train_loss=%.4f, val_age_mae=%.4f, val_gender_acc=%.4f",
            method_name,
            epoch,
            train_loss,
            val_metrics["Age MAE"],
            val_metrics["Gender Accuracy"],
        )

    model.load_state_dict(best_state)
    test_ages, test_pred_ages, test_genders, test_pred_genders, test_races = evaluate_torch_model(
        model, test_loader, device, age_mean, age_std
    )
    return {
        "method": method_name,
        "metrics": metrics_dict(test_ages, test_pred_ages, test_genders, test_pred_genders),
        "fairness": fairness_from_arrays(test_ages, test_pred_ages, test_genders, test_pred_genders, test_races),
        "curve": curve,
    }


def fairness_from_arrays(true_ages, pred_ages, true_genders, pred_genders, races):
    rows = []
    for race_id in sorted(set(races.tolist())):
        mask = races == race_id
        rows.append(
            {
                "Race Group": RACE_NAMES.get(int(race_id), str(int(race_id))),
                "Samples": int(mask.sum()),
                "Age MAE": mean_absolute_error(true_ages[mask], pred_ages[mask]),
                "Gender Accuracy": accuracy_score(true_genders[mask], pred_genders[mask]),
            }
        )
    return rows


def choose_best(results):
    rows = []
    for result in results:
        row = {"method": result["method"], **result["metrics"]}
        rows.append(row)
    age_rank = pd.Series([row["Age MAE"] for row in rows]).rank(method="min", ascending=True)
    acc_rank = pd.Series([row["Gender Accuracy"] for row in rows]).rank(method="min", ascending=False)
    f1_rank = pd.Series([row["Gender F1-score"] for row in rows]).rank(method="min", ascending=False)
    best_index = int((age_rank + acc_rank + f1_rank).idxmin())
    return results[best_index]


def save_main_results(results, path):
    rows = []
    for result in results:
        rows.append({"Method": result["method"], **result["metrics"]})
    pd.DataFrame(rows, columns=["Method", "Age MAE", "Gender Accuracy", "Gender F1-score"]).to_csv(
        path, index=False
    )


def plot_training_curve(results, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    metrics = ["Train Loss", "Validation Age MAE", "Validation Gender Accuracy"]
    for axis, metric in zip(axes, metrics):
        for result in results:
            if result["method"] == "HOG + SVM/SVR":
                continue
            curve = result["curve"]
            epochs = np.arange(1, len(curve[metric]) + 1)
            values = np.asarray(curve[metric], dtype=np.float32)
            if np.all(np.isnan(values)):
                continue
            axis.plot(epochs, values, marker="o", label=result["method"])
        axis.set_ylabel(metric)
        axis.legend()
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_fairness_bar(fairness_rows, path, metric):
    names = [row["Race Group"] for row in fairness_rows]
    values = [row[metric] for row in fairness_rows]
    plt.figure(figsize=(8, 5))
    plt.bar(names, values, color="#4c78a8")
    plt.ylabel(metric)
    plt.xlabel("Race Group")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train all methods and save final result files only.")
    parser.add_argument("--data-dir", default="data/UTKFace")
    parser.add_argument("--labels-csv", default="data/utkface_labels.csv")
    parser.add_argument("--split-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recreate-splits", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hog-image-size", type=int, default=64)
    parser.add_argument("--hog-max-iter", type=int, default=5000)
    parser.add_argument("--fairness-metric", choices=["Age MAE", "Gender Accuracy"], default="Age MAE")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-log", default="experiment_run.log")
    args = parser.parse_args()

    setup_logging(args.run_log)
    set_seed(args.seed)
    datasets = make_datasets(args)
    logging.info("Experiment started with seed=%d cuda_available=%s", args.seed, torch.cuda.is_available())

    results = [
        train_hog_baseline(datasets, args),
        train_deep_model("Custom CNN", "cnn", datasets, args, pretrained=False),
        train_deep_model("ResNet18 Transfer", "resnet18", datasets, args, pretrained=True, freeze_backbone=True),
    ]

    best = choose_best(results)
    save_main_results(results, "main_results.csv")
    pd.DataFrame(best["fairness"], columns=["Race Group", "Samples", "Age MAE", "Gender Accuracy"]).to_csv(
        "fairness_best_model.csv", index=False
    )

    plot_training_curve(results, "training_curve.png")
    plot_fairness_bar(best["fairness"], "fairness_bar.png", args.fairness_metric)
    logging.info("Best model: %s", best["method"])


if __name__ == "__main__":
    main()
