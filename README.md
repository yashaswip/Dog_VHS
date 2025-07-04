
# 🐶 Dog Heart VHS Point Detection

This project uses deep learning to detect 6 anatomical landmarks in dog thoracic X-rays to calculate the **Vertebral Heart Size (VHS)** — a veterinary diagnostic measure for heart enlargement.

## 🚀 Overview

- **Task:** Landmark detection (6 keypoints) → VHS calculation  
- **Model:** EfficientNet-B7 (PyTorch)  
- **Input:** Dog chest radiographs (.png)  
- **Output:** VHS value per image  

## 🧠 Model Details

- **Backbone:** EfficientNet-B7 (pretrained)
- **Loss:** Mean Squared Error (MSE)
- **Optimizer:** Adam  
- **Scheduler:** StepLR

## 📁 Dataset Structure

project/
├── Train/
│ ├── Images/
│ └── Labels/ # .mat files (6 keypoints + VHS)
├── Valid/
│ ├── Images/
│ └── Labels/
├── Test_Images/
│ └── Images/


## ⚙️ Setup & Run

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install efficientnet_pytorch scipy pandas matplotlib

Train the model:

net = VHSNet(pretrained=True).to(device)
...
train_model_v2(net, train_loader, valid_loader, ...)

Run inference:

predict_and_save_vhs(net, test_data, 224, 'test_results.csv')


## 📏 VHS Formula

VHS = 6 × (distance_AB + distance_CD) / distance_EF
AB = Long Axis
CD = Short Axis
EF = Vertebrae reference line

## 📚 References

📄 Research Paper
Youshan Zhang. Regressive Vision Transformer for Dog Cardiomegaly Assessment. Scientific Reports, 14(1):377471128, January 2024.


