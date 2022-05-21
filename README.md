# PyTorch on Apple Silicon

TK - who is this for?

## Setup a machine learning environment with PyTorch on Mac (short version) 

> **Note:** 
> As of May 21 2022, accelerated PyTorch for Mac (PyTorch using the Apple Silicon GPU) is still in beta, so expect some rough edges.

**Requirements:**
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
5. Create a directory to setup PyTorch environment.

```other
mkdir pytorch-test
cd pytorch-test
```

6. Make and activate Conda environment. 

> **Note:** 
> Python 3.8 is the most stable for using the following setup.

```other
conda create --prefix ./env python=3.8
conda activate ./env
```

7. Install the PyTorch nightly version for Mac with pip from the [PyTorch getting started page](https://pytorch.org/get-started/locally/). 


```other
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

This will install the following: 
> Installing collected packages: urllib3, typing-extensions, pillow, numpy, idna, charset-normalizer, certifi, torch, requests, torchvision, torchaudio

8. Install common data science packages.

```other
conda install jupyter pandas numpy matplotlib scikit-learn tqdm 
```

9. Start Jupyter.

```other
jupyter notebook
```

10. Create a new notebook by "New" -> "Notebook: Python 3 (ipykernel)" and run the following code to verfiy all the dependencies are available and check PyTorch version/GPU access.

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

> **Note:**
> See more on running MPS as a backend in the [PyTorch documentation](https://pytorch.org/docs/master/notes/mps.html).

11. To run data/models on an Apple Silicon GPU, use the PyTorch device name `"mps"` with `.to("mps")`. MPS stands for *Metal Performance Shaders*, [Metal is Apple's GPU framework](https://developer.apple.com/metal/). 

```python
import torch

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Create data and send it to the device
x = torch.rand(size=(3, 4)).to(device)
x
```

Finally, you should get something like this:

```
tensor([[2.6020e-01, 9.6467e-01, 7.5282e-01, 1.8063e-01],
        [7.0760e-02, 9.8610e-01, 6.5195e-01, 7.5700e-01],
        [3.4065e-01, 1.8971e-01, 6.0876e-01, 9.3907e-01]], device='mps:0')
```

Congratulations! Your Apple Silicon device is now running PyTorch + a handful of other helpful data science and machine learning libraries.

## TODO
* setup an experiment to run and benchmark a Mac using CPU vs GPU with PyTorch (log the time differences) - this should output a graph with CPU time vs GPU time
* show how to send data/model's to MPS device (if it's available)

## Results

![]("results/apple_m1_pro_TinyVGG_cifar10.png")