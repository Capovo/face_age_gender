import argparse
import csv
import json
from pathlib import Path
import zipfile

from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


DEFAULT_DATASET = "4shL3I/UTKFace"


def get_filename(example, index):
    for key in ("text", "filename", "file_name", "image_id"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).name
    return f"sample_{index:06d}.jpg"


def write_labels_csv_from_zip(archive, output_dir):
    metadata_name = next((name for name in archive.namelist() if name.endswith("metadata.jsonl")), None)
    if metadata_name is None:
        return None

    csv_path = output_dir.parent / "utkface_labels.csv"
    with archive.open(metadata_name) as source, csv_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=["file_name", "age", "gender", "race", "source_name"])
        writer.writeheader()
        for line in source:
            item = json.loads(line.decode("utf-8"))
            source_name = item["text"]
            try:
                age, gender, race = parse_label_from_name(source_name)
            except (IndexError, ValueError):
                continue
            writer.writerow(
                {
                    "file_name": Path(item["file_name"]).name,
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "source_name": source_name,
                }
            )
    return csv_path


def parse_label_from_name(filename):
    parts = Path(filename).stem.split("_")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main():
    parser = argparse.ArgumentParser(description="Download UTKFace from Hugging Face.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/UTKFace")
    parser.add_argument("--limit", type=int, default=None, help="Download only the first N images.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--method", choices=["auto", "zip", "datasets"], default="auto")
    parser.add_argument("--zip-path", default=None, help="Use a local UTKFace zip file instead of downloading.")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming mode.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_path is not None or args.method in ("auto", "zip"):
        zip_path = args.zip_path
        zip_files = []
        if zip_path is None:
            repo_files = list_repo_files(args.dataset, repo_type="dataset")
            zip_files = [name for name in repo_files if name.lower().endswith(".zip")]
            if zip_files:
                zip_path = hf_hub_download(
                    repo_id=args.dataset,
                    filename=zip_files[0],
                    repo_type="dataset",
                )

        if zip_path is not None:
            zip_path = Path(zip_path)
            if not zip_path.exists():
                raise FileNotFoundError(zip_path)

            saved = 0
            skipped = 0
            with zipfile.ZipFile(zip_path) as archive:
                csv_path = write_labels_csv_from_zip(archive, output_dir)
                image_names = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                for index, member_name in enumerate(tqdm(image_names, desc="extracting images")):
                    if args.limit is not None and index >= args.limit:
                        break

                    filename = Path(member_name).name
                    path = output_dir / filename
                    if path.exists() and not args.overwrite:
                        skipped += 1
                        continue

                    with archive.open(member_name) as source, path.open("wb") as target:
                        target.write(source.read())
                    saved += 1

            if csv_path is not None:
                print(f"Labels saved to {csv_path}")
            print(f"Done. saved={saved}, skipped={skipped}, output_dir={output_dir}")
            return

        if args.method == "zip":
            raise RuntimeError(f"No zip file found in dataset repo: {args.dataset}")

    dataset = load_dataset(args.dataset, split=args.split, streaming=not args.no_streaming)
    if args.limit is not None and args.no_streaming:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    saved = 0
    skipped = 0
    for index, example in enumerate(tqdm(dataset, desc="saving images")):
        if args.limit is not None and index >= args.limit:
            break

        image = example["image"]
        filename = get_filename(example, index)
        path = output_dir / filename

        if path.exists() and not args.overwrite:
            skipped += 1
            continue

        image.convert("RGB").save(path)
        saved += 1

    print(f"Done. saved={saved}, skipped={skipped}, output_dir={output_dir}")


if __name__ == "__main__":
    main()
