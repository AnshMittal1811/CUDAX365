wsl --update

# 2. Open Ubuntu shell, update packages
sudo apt-get update
sudo apt-get upgrade -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

sudo apt-get install -y cuda-12-8

echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvidia-smi
nvcc --version       # Should show CUDA 12.x

# Download and run samples
git clone https://github.com/NVIDIA/cuda-samples.git
cd cuda-samples
make -j8
# Then run, e.g.:
./bin/x86_64/linux/release/deviceQuery
./bin/x86_64/linux/release/bandwidthTest
