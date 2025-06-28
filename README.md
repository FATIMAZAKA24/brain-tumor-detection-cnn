# brain-tumor-detection-cnn
**Automated MRI-Based Brain Tumor Classification using Custom Convolutional Neural Networks (CNNs)**

## 💼 Business Use Case
Early and accurate detection of brain tumors is critical for enabling timely diagnosis, treatment planning, and improving patient outcomes. Manual assessment of MRI scans by radiologists is time-consuming and prone to variability. This project leverages deep learning to automate tumor classification in brain MRI images, aiming to provide a decision-support tool for healthcare professionals.

## 📊 Project Summary

This deep learning project builds a **custom CNN model from scratch** to classify brain MRI images into two categories:
- **Yes** → Tumor present
- **No** → No tumor

It also includes a comparative analysis of three different optimization algorithms:
- **Adam**
- **Stochastic Gradient Descent (SGD)**
- **RMSprop**

## 🧾 Dataset

- Source: [Kaggle - Brain Tumor Detection (Br35h 2020)](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection/data)
- 3,060 grayscale MRI images
  - `yes/`: 1500 images with brain tumors
  - `no/`: 1500 images without tumors
- Images resized to **128x128** pixels
- Augmentation applied to training data (rotation, zoom, horizontal flip)


## 🧠 Model Architecture

> A 4-block custom CNN designed and optimized for binary classification

- Convolution + BatchNorm + MaxPooling (4 blocks)
- Increasing filters: 32 → 64 → 128 → 256
- Dropout after each block (20% → 40%)
- Dense layer with 128 units + ReLU
- Final Dense layer with 1 unit + Sigmoid
- Loss function: `binary_crossentropy`
- Activation: `ReLU` for conv layers, `Sigmoid` for final layer


## 🧪 Optimizer Comparison (Results)

| Optimizer | Accuracy | F1 Score | AUC Score | Notes |
|----------|----------|----------|-----------|-------|
| **Adam** | **91%**  | 0.91     | **0.97**  | Best performance, most stable |
| SGD      | 90%      | 0.90     | 0.9569    | Balanced, but less stable |
| RMSprop  | 89%      | 0.89     | 0.971     | Good recall, slightly overfits |

**Adam** was selected as the best optimizer based on all evaluation metrics and learning curve behavior.


## 📈 Evaluation & Visualization

- Metrics: Accuracy, Precision, Recall, F1-score, AUC
- Visualizations:
  - Confusion Matrix
  - Learning Curves (Accuracy & Loss)
  - Optimizer comparison plots
- EarlyStopping applied to avoid overfitting


## 📦 How to Run

> Make sure to install the required dependencies.

```bash
# 1. Clone this repository
git clone https://github.com/FATIMAZAKA24/brain-tumor-detection-cnn.git

# 2. Navigate to the project
cd brain-tumor-detection-cnn

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook
jupyter notebook CNN for Brain Tumour Detection.ipynb
