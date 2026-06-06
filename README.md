# Face Age Gender Project

基于 UTKFace 数据集的人脸年龄回归、性别分类、族裔分类多任务学习项目。

## 项目结构

```text
face_age_gender_project/
├── data/
│   └── UTKFace/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── fairness.py
│   ├── baseline_hog.py
│   ├── download_hf_utkface.py
│   └── app.py
├── environment.yml
├── requirements.txt
└── README.md
```

## 数据准备

把 UTKFace 图片放入：

```text
data/UTKFace/
```

文件名应符合 UTKFace 格式：

```text
age_gender_race_date.jpg
```

其中 `gender` 为 `0=Male, 1=Female`，`race` 为 `0=White, 1=Black, 2=Asian, 3=Indian, 4=Others`。

也可以直接从 Hugging Face 下载：

```bash
python src/download_hf_utkface.py --output-dir data/UTKFace
```

如果 Hugging Face Python 客户端下载较慢，可以先直接下载 zip，再解压并生成标签 CSV：

```bash
curl -L -o data/Data.zip https://huggingface.co/datasets/4shL3I/UTKFace/resolve/main/Data.zip
python src/download_hf_utkface.py --zip-path data/Data.zip --output-dir data/UTKFace
```

只下载少量样本用于测试：

```bash
python src/download_hf_utkface.py --output-dir data/UTKFace --limit 100
```

脚本会从 `metadata.jsonl` 生成：

```text
data/utkface_labels.csv
```

训练和评估代码会自动优先读取这份标签文件。
也可以在命令中显式指定：

```bash
--labels-csv data/utkface_labels.csv
```

## 数据集划分

如果 `data/UTKFace` 下已经存在 `train`、`val`/`valid`、`test` 目录，代码会直接使用已有划分，不移动或改写图片。

如果数据集还没有划分，代码会按固定 seed 自动生成 8:1:1 划分：

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

默认 seed 为 `42`。训练只使用 `train` 和 `val`；最终评估默认只使用 `test`。如需重新生成划分：

```bash
python src/train.py --data-dir data/UTKFace --labels-csv data/utkface_labels.csv --seed 42 --recreate-splits --epochs 0
```

## 安装依赖

使用 Conda 创建环境：

```bash
conda env create -f environment.yml
conda activate face-age-gender
```

如果环境已经创建好，进入项目后直接激活：

```bash
conda activate face-age-gender
```

或使用 pip 安装：

```bash
pip install -r requirements.txt
```

## 训练

传统机器学习 baseline，HOG + LinearSVM 做性别分类，HOG + LinearSVR 做年龄回归：

```bash
python src/baseline_hog.py train --data-dir data/UTKFace --labels-csv data/utkface_labels.csv
```

评估传统 baseline：

```bash
python src/baseline_hog.py evaluate --data-dir data/UTKFace --labels-csv data/utkface_labels.csv --split test --checkpoint checkpoints/hog_baseline.joblib
```

自定义 CNN 多任务模型，同时输出年龄和性别：

```bash
cd face_age_gender_project
python src/train.py --data-dir data/UTKFace --model cnn --epochs 10
```

ResNet18 迁移学习对比模型：

```bash
python src/train.py --data-dir data/UTKFace --labels-csv data/utkface_labels.csv --model resnet18 --epochs 10
```

只训练 ResNet18 任务头、冻结 backbone：

```bash
python src/train.py --data-dir data/UTKFace --labels-csv data/utkface_labels.csv --model resnet18 --freeze-backbone --epochs 10
```

训练后模型默认保存到：

```text
checkpoints/best_model.pt
```

## 评估

```bash
python src/evaluate.py --data-dir data/UTKFace --checkpoint checkpoints/best_model.pt
```

输出指标包括：

- 年龄 MAE
- 性别准确率

## 公平性分析

```bash
python src/fairness.py --data-dir data/UTKFace --checkpoint checkpoints/best_model.pt
```

该脚本会按族裔分组输出年龄 MAE 和性别准确率。

## Streamlit 演示

```bash
streamlit run src/app.py -- --checkpoint checkpoints/best_model.pt
```

上传单张人脸图片后，页面会显示年龄、性别和族裔预测结果。
