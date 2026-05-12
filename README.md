# InformalMambaNet

### Semantic–State Decoupled State-Space Architecture for Informal Settlement Segmentation in Satellite Imagery

---

# 1. Description

This repository contains the implementation of the research work entitled:

**“InformalMambaNet: Semantic–State Decoupled State-Space Architecture for Informal Settlement Segmentation in Satellite Imagery.”**

The proposed framework introduces a lightweight segmentation architecture called **InformalMambaNet**, designed specifically for satellite imagery segmentation in informal settlements.

The model is based on a novel concept called:

* **Semantic–State Decoupled Spatial Reasoning (SSDSR)**

The framework uses:

* A lightweight CNN semantic encoder
* Dual-Scale State Interaction (DSSI)
* Boundary-aware spatial state propagation
* State-aware decoder refinement

The model was evaluated on high-resolution satellite imagery for semantic segmentation of informal settlements.

The task is to classify pixels into the following categories:

1. **Slums**
2. **Built-Up**
3. **Roads**
4. **Vegetation**
5. **Barren**
6. **Water**
7. **Background**

---

# 2. System Requirements

* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
* **Python:** 3.9 – 3.11
* **GPU (Recommended):** NVIDIA GPU with CUDA 11.6+
* **RAM:** Minimum 8 GB (16 GB recommended)
* **Storage:** At least 15 GB free

---

# 3. Required Libraries

Install dependencies using:

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```text
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
opencv-python>=4.6.0
Pillow>=9.3.0
albumentations>=1.3.0
einops>=0.6.0
```

---

# 4. Usage Instructions

## 1. Clone the Repository

```bash
git clone https://github.com/Chanduchawla/informalmambanet
cd informalmambanet
```

---

## 2. Set Up the Dataset

Organize the dataset in the following structure:

```text
dataset/
 ├── train_images/
 │   └── train/
 ├── train_masks/
 │   └── train/
 ├── val_images/
 │   └── val/
 ├── val_masks/
 │   └── val/
 ├── test_images/
 │   └── test/
 └── test_masks/
     └── test/
```

Dataset:

```bash
https://www.kaggle.com/datasets/ayushdabra/sdsa-dse-406-606-demo-data
```

---

## 3. Run the Project

### Run as Python Script

```bash
python project.py
```

---

## 4. Run in Jupyter Notebook

You can also run the project in Jupyter Notebook.

### Option 1: Open `.py` directly

* Open Jupyter Notebook
* Open the `.py` file directly

### Option 2: Convert `.py` to `.ipynb`

```bash
jupyter nbconvert --to notebook project.py --output project.ipynb
```

---

# 5. Model Architecture

The proposed InformalMambaNet framework contains:

* CNN Semantic Stem
* Boundary-Aware State Scan
* Dual-Scale State Interaction (DSSI)
* State-Aware Decoder

The architecture follows the sequence:

```text
Semantic Abstraction
        ↓
State Propagation
        ↓
Boundary Refinement
```

This prevents noisy state diffusion during early training.

---

# 6. Experiments

The repository includes experiments with:

* DeepLabV3+
* SegFormer
* InformalMambaNet v1
* InformalMambaNet v2
* InformalMambaNet v3 (Proposed)

---

# 7. Results

| Model               | Parameters | Validation mIoU | Validation Dice |
| ------------------- | ---------- | --------------- | --------------- |
| DeepLabV3+          | 41M        | 0.57            | 0.63            |
| SegFormer           | 25M        | 0.38            | 0.45            |
| InformalMambaNet v1 | 195K       | 0.29            | 0.31            |
| InformalMambaNet v2 | 215K       | 0.39            | 0.44            |
| InformalMambaNet v3 | 247K       | 0.60            | 0.67            |



---

# 8. Output

The framework produces:

* ✅ Saved model weights
* 📈 Performance metrics (mIoU, Dice Score, Precision, Recall)
* 📉 Training history plots
* 🖼 Predicted segmentation masks
* 📊 Qualitative comparison visualizations

---

# 9. Reference

If you use this code or research in your work, please cite:

**Chevala Chandu, Ashish Kumar Sahu**

*InformalMambaNet: Semantic–State Decoupled State-Space Architecture for Informal Settlement Segmentation in Satellite Imagery.*

---

# 10. Contact

For queries, collaborations, or clarifications:

👤 **Chevala Chandu**
📧 Email: [chevalachandu93@gmail.com](mailto:chevalachandu93@gmail.com)

GitHub Profile:
[ChanduChawla GitHub](https://github.com/Chanduchawla)

---

# 11. License

This project is intended for academic and research purposes.
