# DI725_termproject_2030336

The final project of DI 725: Transformers and Attention-Based Deep Networks. There will
be three phases throughout this project. The first phase will include the preparation of a brief literature survey and a
project proposal. The second phase will cover the preliminary results and benchmarking. The third phase will conclude
the project with results and comparisons.
There will be a base vision-language foundation model (PaliGemma) that is suitable for various tasks including but
not limited to visual question answering, image captioning and object detection. Our task is to generate image
captions. We will structure our own research question, and propose a project to build on top of this foundation.

## Setup
1. Local Setup:
   - Set execution policy: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
   - Create virtual environment: `C:\Users\nesil.bor\AppData\Local\Microsoft\WindowsApps\python3.12.exe -m venv .venv`
   - Activate: `.\.venv\Scripts\Activate.ps1`
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure >10GB free disk space on C: drive
   - Run training: `python source/train_lora.py`
2. Colab Setup (Recommended):
   - Upload project to Google Drive
   - Authenticate with Hugging Face token for gated repos
   - Use Colab notebook with T4 GPU runtime
   - Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   - Use batch processing and gradient accumulation
   - Run training and download model
Note: Reduced max_train_samples to 500 to avoid CUDA out of memory on T4 GPU.