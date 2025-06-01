# hel_triple_ride_detection
# 🚀 mini_interface: YOLOv8 Object Detection Web App

This project is a **web-based object detection tool** built using [Streamlit](https://streamlit.io/) and [YOLOv8](https://docs.ultralytics.com/). It allows users to upload **images or videos** and detects **only the object with the highest confidence score** per frame using a custom-trained YOLOv8 model.

---

## 📌 Project Highlights

- 🔍 Object detection in images and videos
- 🏆 Only displays the **top-confidence detection**
- 📤 File uploader with support for `.jpg`, `.png`, `.mp4`
- 🎯 Confidence threshold slider
- 🌐 Custom theme and styling with Streamlit
- ⚙️ Powered by a custom-trained YOLOv8 model (`best3.pt`)

---

## 🧠 Model Training Overview

The YOLOv8 model was trained using a custom dataset for detecting:
- Helmet usage
- Triple-riding on motorcycles

Training pipeline included:
- Dataset split (train/val/test)
- Augmentations: Mosaic, MixUp, Flips, HSV, Scale
- Validation with mAP, precision, recall

Model file: `best3.pt` (stored externally on Google Drive)

---

## 📂 Project Structure

