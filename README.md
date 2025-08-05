<h1>Point-SAM: Segment Anything in 3D Point Clouds for Multiple Objects</h1>

Point-SAM is an extended version of the Segment Anything Model (SAM) designed for 3D point cloud data. 
It enables zero-shot segmentation of multiple objects in complex scenes, supporting RGB and XYZ inputs, and can generate precise object masks without additional training.

## 🙏 Reference

-[Point-SAM](https://github.com/zyc00/Point-SAM/tree/main)
-[Point-SAM model.safetensor](https://huggingface.co/yuchen0187/Point-SAM/tree/main)

## 📦 Installation

1. Conda 환경 생성 및 활성화

```bash

conda create -n point-sam python=3.10 -y
conda activate point-sam
```
⚠️ Python ≥3.8만 요구하지만, PyTorch 2.1+의 안정성과 호환성을 위해 3.10 추천

2. 필수 패키지 설치(PyTorch, TorchVision, timm)

```bash
# PyTorch 2.1.0 + torchvision 0.16.0 (CUDA 12.1 기준)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# timm >= 0.9.0
pip install timm>=0.9.0
```
💡 cu121은 RTX 40 시리즈에 최적화된 CUDA 12.1을 의미함. 내 GPU (RTX 4060 Laptop)는 CUDA 12.1 이상에서 성능 최적화됨!

3. g++ 9.3.0 설치 (Apex, Torkit3D 빌드용)

```bash
conda install -c conda-forge gxx_linux-64=9.3.0 -y
```

4. Point-SAM Clone & Submodule

```bash
git clone https://github.com/zyc00/Point-SAM.git
cd Point-SAM
git submodule update --init --recursive
```

5. CUDA 12.1 toolkit 저장소 추가 및 패키지 다운로드

```bash
# 저장소 PIN 설정
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# CUDA 12.1.1 local installer (.deb)
wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/

# 리포지터리 업데이트 & 설치
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

6. 설치 확인

```bash
ls /usr/local/ | grep cuda-12.1

/usr/local/cuda-12.1/bin/nvcc --version
cuda-12.1
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```

7. 환경변수 설정(conda 설정용)

```bash
mkdir -p ~/.conda/envs/point-sam/etc/conda/activate.d
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.conda/envs/point-sam/etc/conda/activate.d/env_vars.sh
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.conda/envs/point-sam/etc/conda/activate.d/env_vars.sh
echo 'export CMAKE_PREFIX_PATH=$CUDA_HOME:$CMAKE_PREFIX_PATH' >> ~/.conda/envs/point-sam/etc/conda/activate.d/env_vars.sh
```

8. Torkit3D 설치

```bash
git submodule update --init third_party/torkit3d
FORCE_CUDA=1 pip install third_party/torkit3d
```

9. 나머지 패키지 설치

```bash
pip install hydra-core omegaconf plyfile open3d einops
```

## 🏗️ Run

```bash
 python evaluation/inference_id.py 
--input_path demo/static/models/250722_two_chairs.ply 
--ckpt_path pretrained/model.safetensors
```
