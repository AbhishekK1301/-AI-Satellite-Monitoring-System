🌍 AI Disaster & Agriculture Monitoring System
Single-File Deep Learning Project (ai_disaster_system.py)

This project is an end-to-end AI system for analyzing satellite and drone images to detect natural disasters (Floods, Fires, Drought) and monitor agricultural conditions.
It provides image classification, forecasting, real-time alerts, and an interactive dashboard, all inside a single Python file: ai_disaster_system.py.

🚀 Features
🔍 1. Disaster Detection (ResNet18 – Transfer Learning)

Uses ResNet18 for high-accuracy image classification

Detects:
Flood
Fire
Drought
Healthy vegetation

Fully ImageFolder-compatible dataset loading

📈 2. LSTM-Based Disaster Forecasting

A small LSTM model predicts time-series patterns such as:

Flood rise
Fire spread
Drought progression

(Provided as a demo module.)

📡 3. Alert System (Email + SMS)

Send email alerts via SMTP

Send SMS alerts via Twilio (placeholder included)

Alerts auto-trigger for high-risk detections

🖥️ 4. Streamlit Dashboard

An interactive UI to:

Upload satellite images

Run AI detection

Display predictions + confidence

Trigger alerts directly from dashboard

Launch using:

streamlit run ai_disaster_system.py -- --mode serve

🧱 5. Single-File Architecture

Everything is contained inside one file:

Preprocessing

Dataloaders

CNN model

LSTM predictor

Training & inference modules

Dashboard

Alerts

Makes it ideal for students, demonstrations, project submissions, and quick deployment.
