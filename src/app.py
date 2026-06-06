import argparse

from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from dataset import GENDER_NAMES
from model import build_model


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint.get("model_name", "cnn"),
        pretrained=False,
        freeze_backbone=checkpoint.get("freeze_backbone", False),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint.get("age_mean", 0.0), checkpoint.get("age_std", 1.0)


def preprocess(image):
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    st.set_page_config(page_title="Face Age Gender Demo", layout="centered")
    st.title("Face Age / Gender / Race Demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to run prediction.")
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input image", use_container_width=True)

    try:
        model, age_mean, age_std = load_model(args.checkpoint, device)
    except FileNotFoundError:
        st.error(f"Checkpoint not found: {args.checkpoint}")
        return

    with torch.no_grad():
        tensor = preprocess(image).to(device)
        pred_age, pred_gender = model(tensor)
        age = max(0, round(pred_age.item() * age_std + age_mean, 1))
        gender = GENDER_NAMES.get(pred_gender.argmax(dim=1).item(), "Unknown")

    col1, col2 = st.columns(2)
    col1.metric("Age", age)
    col2.metric("Gender", gender)


if __name__ == "__main__":
    main()
