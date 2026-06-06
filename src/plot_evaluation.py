import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = ["Age MAE", "Gender Accuracy", "Gender F1-score"]


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def plot_main_results(results, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = ["#4c78a8", "#59a14f", "#e15759"]

    for axis, metric, color in zip(axes, METRICS, colors):
        axis.bar(results["Method"], results[metric], color=color)
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot model evaluation comparison from main_results.csv.")
    parser.add_argument("--results", default="main_results.csv")
    parser.add_argument("--output", default="evaluation_comparison.png")
    parser.add_argument("--log-file", default="evaluation_run.log")
    args = parser.parse_args()

    setup_logging(args.log_file)
    results_path = Path(args.results)
    output_path = Path(args.output)

    logging.info("Evaluation plotting started")
    logging.info("Reading results from %s", results_path)
    results = pd.read_csv(results_path)

    missing = [column for column in ["Method", *METRICS] if column not in results.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for _, row in results.iterrows():
        logging.info(
            "%s: Age MAE=%.4f, Gender Accuracy=%.4f, Gender F1-score=%.4f",
            row["Method"],
            row["Age MAE"],
            row["Gender Accuracy"],
            row["Gender F1-score"],
        )

    best_age = results.loc[results["Age MAE"].idxmin(), "Method"]
    best_acc = results.loc[results["Gender Accuracy"].idxmax(), "Method"]
    best_f1 = results.loc[results["Gender F1-score"].idxmax(), "Method"]
    logging.info("Best Age MAE model: %s", best_age)
    logging.info("Best Gender Accuracy model: %s", best_acc)
    logging.info("Best Gender F1-score model: %s", best_f1)

    plot_main_results(results, output_path)
    logging.info("Saved evaluation comparison chart to %s", output_path)


if __name__ == "__main__":
    main()
