# ai_disaster_system.py
"""
AI Disaster & Agriculture Monitoring - single-file implementation

Features:
- Data loader (ImageFolder-compatible)
- ResNet18-based detector (transfer learning)
- Training & evaluation loop
- Simple LSTM predictor for time-series (toy)
- Alert functions (SMTP + Twilio placeholders)
- Streamlit dashboard for inference

Usage examples:
    python ai_disaster_system.py --mode train --epochs 10 --data_dir data --model_out models/detector.pth
    python ai_disaster_system.py --mode infer --image example.jpg --model models/detector.pth
    streamlit run ai_disaster_system.py -- --mode serve --model models/detector.pth

Author: Generated for user
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

# Optional imports for alerts
import smtplib
from email.message import EmailMessage

# Twilio placeholder (uncomment if you want to use it and have credentials)
# from twilio.rest import Client

# ---------------------------
# Utility helpers
# ---------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def load_image_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

# ---------------------------
# Config / defaults
# ---------------------------
DEFAULTS = {
    "img_size": 224,
    "batch_size": 16,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_classes": 4,  # change to actual number of classes in your dataset
    "labels": ["Flood", "Fire", "Drought", "Healthy"],
    "models_dir": "models",
    "logs_dir": "logs",
    "data_dir": "data"
}

ensure_dir(DEFAULTS["models_dir"])
ensure_dir(DEFAULTS["logs_dir"])

# ---------------------------
# Data: DataLoader builder
# ---------------------------
def build_dataloaders(data_dir: str,
                      img_size: int = 224,
                      batch_size: int = 16,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Expects:
      data_dir/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg

    returns train_loader, val_loader, class_names
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = train_ds.classes
    return train_loader, val_loader, class_names

# ---------------------------
# Model: Detector (ResNet18 transfer learning)
# ---------------------------
def build_detector(n_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(pretrained=pretrained)
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_classes)
    return model

# ---------------------------
# LSTM Predictor (toy)
# ---------------------------
class SimpleLSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, out_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---------------------------
# Training & Eval
# ---------------------------
def train_detector(model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   device: str,
                   epochs: int = 5,
                   lr: float = 1e-4,
                   model_out: str = "models/detector.pth"):
    device = torch.device(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        # Validation
        val_loss, val_acc, _ = evaluate_detector(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - t0
        print(f"Epoch {epoch}/{epochs} - {epoch_time:.1f}s - train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": getattr(train_loader.dataset, "classes", None)
            }, model_out)
            print(f"Saved best model to {model_out} (val_acc {best_val_acc:.4f})")

    # Save history
    ensure_dir(os.path.dirname(model_out) or ".")
    hist_path = os.path.join(DEFAULTS["logs_dir"], f"history_{int(time.time())}.json")
    save_json(history, hist_path)
    print("Training complete. History saved to", hist_path)
    return history

def evaluate_detector(model: nn.Module, loader: DataLoader, device: str):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    val_loss = running_loss / len(loader.dataset)
    val_acc = correct / total
    report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
    return val_loss, val_acc, report

# ---------------------------
# Inference
# ---------------------------
def load_model_for_inference(model_path: str, device: str):
    device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device)
    # attempt to rebuild model automatically if possible
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
        # detect num classes from final layer size
        # we assume ResNet-like architecture
        # best-effort: if shapes match resnet18 final layer, rebuild accordingly
        # default to DEFAULTS["num_classes"] if unknown
        try:
            out_features = list(state.keys())[-1]  # not reliable, but try
        except Exception:
            out_features = None

        # instantiate ResNet18 and load state
        model = build_detector(DEFAULTS["num_classes"])
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        class_names = checkpoint.get("class_names", DEFAULTS["labels"])
        return model, class_names
    else:
        # older save: raw state_dict
        model = build_detector(DEFAULTS["num_classes"])
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model, DEFAULTS["labels"]

def predict_image(model: nn.Module, image_path: str, class_names: List[str], device: str, img_size=224):
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    img = load_image_pil(image_path)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return {
            "pred_class": class_names[pred.item()] if class_names and pred.item() < len(class_names) else str(pred.item()),
            "confidence": float(conf.item())
        }

# ---------------------------
# Alerts (simple)
# ---------------------------
def send_email_alert(smtp_server: str,
                     smtp_port: int,
                     username: str,
                     password: str,
                     subject: str,
                     body: str,
                     to_emails: List[str]):
    """
    Simple SMTP email sending (Gmail might require app password).
    """
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    with smtplib.SMTP(smtp_server, smtp_port) as s:
        s.starttls()
        s.login(username, password)
        s.send_message(msg)
    print("Email alert sent to", to_emails)

def send_sms_alert_twilio(account_sid: str, auth_token: str, from_number: str, to_number: str, message: str):
    """
    Placeholder for Twilio SMS sending. Uncomment Twilio import & install if using.
    """
    # client = Client(account_sid, auth_token)
    # client.messages.create(body=message, from_=from_number, to=to_number)
    print(f"[SIMULATED SMS] To: {to_number} From: {from_number} Msg: {message}")

# ---------------------------
# Streamlit dashboard (simple)
# ---------------------------
def run_streamlit_app(model_path: str, device: str, img_size: int = 224):
    import streamlit as st

    st.set_page_config(page_title="AI Satellite Disaster Monitor", layout="centered")
    st.title("🌍 AI Satellite Disaster Detection")

    model, class_names = load_model_for_inference(model_path, device)

    uploaded = st.file_uploader("Upload a satellite/drone image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        # save to temp
        temp_path = os.path.join("temp_uploaded.jpg")
        img.save(temp_path)
        if st.button("Run Detection"):
            with st.spinner("Predicting..."):
                out = predict_image(model, temp_path, class_names, device, img_size=img_size)
                st.markdown(f"**Prediction:** {out['pred_class']}")
                st.markdown(f"**Confidence:** {out['confidence']*100:.2f}%")
                # Example: if serious event detected, show alert button
                if out['pred_class'] in ["Flood", "Fire", "Drought"]:
                    st.warning("Critical event detected! Consider sending alerts.")
                    if st.button("Send Test Email Alert"):
                        # NOTE: In production, do not hardcode credentials. Use secrets.
                        st.info("Simulating email alert (configure SMTP settings to send real email).")
                        try:
                            send_email_alert("smtp.example.com", 587, "sender@example.com", "password",
                                             f"Alert: {out['pred_class']} detected",
                                             f"Detected {out['pred_class']} with confidence {out['confidence']:.2f}",
                                             ["recipient@example.com"])
                            st.success("Email alert sent (simulated).")
                        except Exception as e:
                            st.error("Failed to send email (expected if not configured). " + str(e))
    st.markdown("---")
    st.markdown("**Model path:** " + model_path)
    st.markdown("**Device:** " + device)

# ---------------------------
# Toy: Sequence forecasting demo
# ---------------------------
def demo_lstm_predictor():
    # toy example: trend + noise
    seq_len = 20
    data = np.cumsum(np.random.randn(500) * 0.1 + 0.05)
    # create sequences
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X = np.array(X)[:, :, None].astype(np.float32)
    y = np.array(y).astype(np.float32)[:, None]

    # split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    device = DEFAULTS["device"]
    model = SimpleLSTMPredictor(input_size=1, hidden_size=32, num_layers=1, out_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    epochs = 50
    for ep in range(1, epochs+1):
        model.train()
        idx = np.random.permutation(len(X_train))
        losses = []
        for i in range(0, len(X_train), 32):
            batch_idx = idx[i:i+32]
            xb = torch.tensor(X_train[batch_idx]).to(device)
            yb = torch.tensor(y_train[batch_idx]).to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if ep % 10 == 0:
            print(f"[LSTM] Epoch {ep}/{epochs} loss {np.mean(losses):.4f}")
    print("LSTM demo complete. Predicting next 5 steps...")
    model.eval()
    seq = X_val[-1:,:,:]  # last sequence
    preds = []
    with torch.no_grad():
        for _ in range(5):
            p = model(torch.tensor(seq).to(device)).cpu().numpy().ravel()[0]
            preds.append(float(p))
            # shift seq and append predicted
            seq = np.concatenate([seq[:,1:,:], np.array([[[p]]])], axis=1)
    print("Next 5 predictions:", preds)

# ---------------------------
# Main CLI
# ---------------------------
def main_cli(args):
    mode = args.mode
    device = args.device or DEFAULTS["device"]

    if mode == "train":
        data_dir = args.data_dir or DEFAULTS["data_dir"]
        batch_size = args.batch_size or DEFAULTS["batch_size"]
        img_size = args.img_size or DEFAULTS["img_size"]
        epochs = args.epochs or 5
        model_out = args.model_out or os.path.join(DEFAULTS["models_dir"], "detector.pth")
        print("Building dataloaders from", data_dir)
        train_loader, val_loader, class_names = build_dataloaders(data_dir, img_size=img_size, batch_size=batch_size, num_workers=args.num_workers or DEFAULTS["num_workers"])
        print("Classes:", class_names)
        model = build_detector(len(class_names))
        history = train_detector(model, train_loader, val_loader, device=device, epochs=epochs, lr=args.lr or 1e-4, model_out=model_out)
        print("Finished training. Model saved to:", model_out)
    elif mode == "infer":
        image_path = args.image
        model_path = args.model or os.path.join(DEFAULTS["models_dir"], "detector.pth")
        if not image_path:
            print("Please provide --image path/to/image.jpg")
            return
        print("Loading model", model_path)
        model, class_names = load_model_for_inference(model_path, device)
        out = predict_image(model, image_path, class_names, device, img_size=args.img_size or DEFAULTS["img_size"])
        print("Prediction:", out)
        # optional alert logic
        if out['pred_class'] in ["Flood", "Fire", "Drought"] and out['confidence'] > 0.6:
            print("High-confidence critical event detected. You may want to send alerts.")
    elif mode == "serve":
        model_path = args.model or os.path.join(DEFAULTS["models_dir"], "detector.pth")
        # Run streamlit app
        print("Starting Streamlit app with model", model_path)
        run_streamlit_app(model_path, device, img_size=args.img_size or DEFAULTS["img_size"])
    elif mode == "lstm_demo":
        demo_lstm_predictor()
    else:
        print("Unknown mode. Choose from: train, infer, serve, lstm_demo")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Disaster & Agri Monitoring single-file tool")
    parser.add_argument("--mode", type=str, default="serve", help="train | infer | serve | lstm_demo")
    parser.add_argument("--data_dir", type=str, default=DEFAULTS["data_dir"])
    parser.add_argument("--model_out", type=str, default=os.path.join(DEFAULTS["models_dir"], "detector.pth"))
    parser.add_argument("--model", type=str, default=os.path.join(DEFAULTS["models_dir"], "detector.pth"))
    parser.add_argument("--image", type=str, help="Path to image for inference")
    parser.add_argument("--epochs", type=int, help="Number of epochs (train)")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--img_size", type=int, help="Image size (square)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--device", type=str, default=DEFAULTS["device"], help="cpu | cuda")
    parser.add_argument("--num_workers", type=int, default=DEFAULTS["num_workers"])
    args, unknown = parser.parse_known_args()

    # When running with streamlit, streamlit passes extra args; we accept them.
    main_cli(args)
