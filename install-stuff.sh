#!/bin/bash
set -e

source venv/bin/activate

if [[ ! -f cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb ]] ; then
  wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
  sudo dpkg -i cudss-local-repo-ubuntu2404-0.7.1_0.7.1-1_arm64.deb
  sudo cp /var/cudss-local-repo-ubuntu2404-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
fi

if [[ ! -f nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb ]] ; then
  wget https://developer.download.nvidia.com/compute/nvpl/25.5/local_installers/nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb
  sudo dpkg -i nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb
  sudo cp /var/nvpl-local-repo-ubuntu2404-25.5/nvpl-*-keyring.gpg /usr/share/keyrings/
fi

sudo apt update
sudo apt install nvpl libnvpl-tensor-dev libcudss0-cuda-13

# Install PyTorch + TorchVision for Python 3.12 (JetPack 7 / CUDA 13.0+)
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130

# 1. Force-align the core dependencies to what TensorRT-Edge-LLM expects.
# (We use --force-reinstall to downgrade the 'too new' packages)
#pip install --force-reinstall \
#python3 -c "import onnx; print(f'ONNX Version: {onnx.__version__}'); print(f'FP4 Support: {hasattr(onnx.TensorProto, 'FLOAT4E2M1')}')"
#python3 -c "import onnx; print(f'ONNX Version: {onnx.__version__}'); print(f'FP4 Support: {hasattr(onnx.TensorProto, 'FLOAT4E2M1')}')"
onnyx_version="$(python3 -c "import onnx; import modelopt; print(f'{onnx.__version__}')")"

if [[ "${onnyx_version}" != "1.19.0" ]] ; then
  echo pip install \
    "onnx==1.19.0" \
    "nvidia-modelopt[onnx]==0.39.0" \
    "nvidia-modelopt[torch]==0.39.0" \
    "numpy~=2.2.6"
fi

# 2. Re-install GraphSurgeon (Must be done AFTER onnx to link correctly)
#pip install --force-reinstall 'git+https://github.com/NVIDIA/TensorRT.git@main#egg=onnx-graphsurgeon&subdirectory=tools/onnx-graphsurgeon'

#pip install --force-reinstall --no-deps 'git+https://github.com/NVIDIA/TensorRT.git@main#egg=onnx-graphsurgeon&subdirectory=tools/onnx-graphsurgeon'

modelopt_version="$(python3 -c "import onnx; import modelopt; print(f'{modelopt.__version__}')")"

if [[ "${modelopt_version}" != "0.39.0" ]] ; then
  pip install --no-deps 'git+https://github.com/NVIDIA/TensorRT.git@main#egg=onnx-graphsurgeon&subdirectory=tools/onnx-graphsurgeon'
fi

python3 -c "import torch; print(f'Torch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
