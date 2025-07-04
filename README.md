# 🐶 Dog Heart VHS Point Detection

This project uses deep learning to detect 6 anatomical landmarks in dog thoracic X-rays to calculate the **Vertebral Heart Size (VHS)** — a veterinary diagnostic measure for heart enlargement.

---

## 🚀 Overview

- **Task:** Landmark detection (6 keypoints) → VHS calculation  
- **Model:** EfficientNet-B7 (PyTorch)  
- **Input:** Dog chest radiographs (`.png`)  
- **Output:** VHS value per image  

---

## 🧠 Model Details

- **Backbone:** EfficientNet-B7 (pretrained)  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Scheduler:** StepLR  

---

## 📁 Dataset Structure
```bash
project/
├── Train/
│ ├── Images/
│ └── Labels/ # .mat files with 6 keypoints + VHS
├── Valid/
│ ├── Images/
│ └── Labels/
├── Test_Images/
│ └── Images/
```
---

## ⚙️ Setup & Run

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install efficientnet_pytorch scipy pandas matplotlib
```

### 2. Train the Model
```bash
net = VHSNet(pretrained=True).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

train_loss, valid_loss, valid_acc = train_model_v2(
    net, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs=100
)

torch.save(net.state_dict(), 'final_model_v2.pth')
```
### 3. Run Inference
```bash
test_data = CustomTestDataset('/content/Test_Images', build_transforms(224))
predict_and_save_vhs(net, test_data, 224, 'test_results.csv')
Sample Output (test_results.csv):

ImageName,VHS
dog001.png,10.28
dog002.png,9.95
...
```
📏 VHS Formula
```bash

VHS = 6 × (distance_AB + distance_CD) / distance_EF
```
Where:

A–B → Long Axis.


C–D → Short Axis.


E–F → Vertebral (spinal) reference line.

📚 References

📄 Dog Heart Vertebral Heart Size Point Detection – ResearchGate

Youshan Zhang. Regressive Vision Transformer for Dog Cardiomegaly Assessment. Scientific Reports, 14(1):377471128, January 2024.

---
