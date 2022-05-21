# PyTorch on Apple Silicon

TK - who is this for?

## Setup a machine learning environment with PyTorch on Mac (short version) 

TK - note: as of XXX this is still beta

Requirements:
* Apple Silicon Mac (M1, M1 Pro, M1 Max, M1 Ultra, etc).
* macOS 12.3+ (PyTorch will work on previous versions but the GPU on your Mac won't get used, this means slower code).

### Steps

1. Download and install Homebrew from [https://brew.sh](https://brew.sh). Follow the steps it prompts you to go through after installation.
2. [Download Miniforge3](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh) (Conda installer) for macOS arm64 chips (M1, M1 Pro, M1 Max, M1 Ultra).
3. Install Miniforge3 into home directory.

```other
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

4. Restart terminal.
5. Create a directory to setup TensorFlow environment.

```other
mkdir tensorflow-test
cd tensorflow-test
```

6. Make and activate Conda environment. **Note:** Python 3.8 is the most stable for using the following setup.

```other
conda create --prefix ./env python=3.8
conda activate ./env
```

7. Install the PyTorch nightly version for Mac with pip from the [PyTorch getting started page](https://pytorch.org/get-started/locally/). **Note:** As of May 21 2022, PyTorch on Mac is still in beta, so expect some rough edges.

```other
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

This will install the following: 
> Installing collected packages: urllib3, typing-extensions, pillow, numpy, idna, charset-normalizer, certifi, torch, requests, torchvision, torchaudio

8. Install common data science packages.

```other
conda install jupyter pandas numpy matplotlib scikit-learn tqdm 
```

9. Start Jupyter Notebook.

```other
jupyter notebook
```

10. Import dependencies and check PyTorch version/GPU access.

```python
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
```


If it all worked, you should see something like:

```other
PyTorch version: 1.12.0.dev20220519
Is MPS (Metal Performance Shader) built? True
Is MPS available? True
Using device: mps
```

11. To send data/models to the `"mps"` device, use `.to("mps")`.

```python
import torch

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Create data and send it to the device
x = torch.rand(size=(3, 4)).to(device)
```

## TODO
* setup an experiment to run and benchmark a Mac using CPU vs GPU with PyTorch (log the time differences) - this should output a graph with CPU time vs GPU time
* show how to send data/model's to MPS device (if it's available)

## Results

![]("results/apple_m1_pro_TinyVGG_cifar10.png")