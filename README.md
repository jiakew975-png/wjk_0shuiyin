# wjk_0shuiyin
A zero watermarking method based on image contour (neural network, code)
#ExCNN and ExCNNG are neural network codes; Arnold, Narnold obfuscation：reverse obfuscation code; NewCanny Canny operator code;
#Various attack codes are included in the attack

# ExCNN、ExCNNG Training Environment

This project implements the **ExCNN residual encoder-decoder network** for contour/edge feature learning, with training, evaluation and image reconstruction scripts.

---

## ⚙️ Environment

- OS: Windows 10 / Ubuntu 20.04
- Python: 3.8+
- CUDA: 11.3 (optional, if using GPU)
- GPU: NVIDIA RTX series (recommended), but CPU is also supported

### Python Dependencies
All required packages are listed below:

- torch >= 1.12.0
- torchvision >= 0.13.0
- numpy >= 1.21.0
- pillow >= 9.0.0
- matplotlib >= 3.5.0
- opencv-python >= 4.5.0

---

## 📦 Installation

```bash
# 1. Clone repository
git clone https://github.com/YourUsername/ExCNN-ContourSeg.git
cd ExCNN-ContourSeg

# 2. Create virtual environment
conda create -n excnn python=3.8 -y
conda activate excnn

# 3. Install PyTorch (GPU version if CUDA available)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 4. Install other dependencies
pip install numpy pillow matplotlib opencv-python
🖼 Example Dataset Structure
grouped_train2000_trans0.5/
    └── GBarbara/
        ├── img1.png
        ├── img2.png
        └── ...
a/
    └── a.png   # target image (ground truth)

🚀 Training
python train.py


The best checkpoint will be saved in saved_822_GBarbara/excnn_best.pth

Final model will be saved in saved_822_GBarbara/excnn_final.pth

🔍 Testing
python test.py


The reconstructed images will be saved in:

result822/GBarbara/

