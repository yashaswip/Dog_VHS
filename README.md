🐶 Dog Heart Vertebral Heart Size (VHS) Point Detection

This project uses deep learning to detect six anatomical landmarks in thoracic X-ray images of dogs and compute the Vertebral Heart Size (VHS) — a diagnostic tool used by veterinarians to detect heart enlargement.

🚀 Project Overview

Goal: Predict 6 anatomical landmark points on dog chest radiographs.

Output: VHS (Vertebral Heart Size) measurement.

Model: EfficientNet-B7 (PyTorch)

Paper Reference:Dog Heart Vertebral Heart Size Point Detection – ResearchGate

🧠 Model Architecture

Backbone: EfficientNet-B7 (pretrained on ImageNet)

Output Layer: Linear layer with 12 outputs (x, y for 6 points)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Scheduler: StepLR

🛠️ Setup Instructions

1. Install dependencies

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install efficientnet_pytorch scipy pandas matplotlib

2. Prepare your dataset structure

project/
├── Train/
│   ├── Images/
│   └── Labels/  (.mat files with 6 keypoints & VHS)
├── Valid/
│   ├── Images/
│   └── Labels/
├── Test_Images/
│   └── Images/

⚙️ Training the Model

net = VHSNet(pretrained=True).to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

train_loss, valid_loss, valid_acc = train_model_v2(
    net, train_loader, valid_loader, loss_fn, optimizer, scheduler, epochs=100
)

torch.save(net.state_dict(), 'final_model_v2.pth')

🧪 Inference & VHS Calculation

test_data = CustomTestDataset('/content/Test_Images', build_transforms(224))
predict_and_save_vhs(net, test_data, 224, 'test_results_v2.csv')

Sample Output (test_results_v2.csv):

ImageName

VHS

dog001.png

10.28

dog002.png

9.95

🧲 VHS Calculation Formula



Where:

A–B → Long Axis

C–D → Short Axis

E–F → Spine line

📊 Sample Training Logs

Epoch 1: Training Loss = 0.0342, Validation Loss = 0.0317
Epoch 2: Training Loss = 0.0201, Validation Loss = 0.0189
...

📁 Model Download Link

Download Final Model

📚 Citations

Zhang, Qoushan."Regressive Vision Transformer for Dog Cardiomegaly Assessment."Scientific Reports, 14(1):377471128, January 2024.


