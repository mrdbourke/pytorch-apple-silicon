# PyTorch on Apple Silicon

**You:** Have an Apple Silicon Mac (M1, M1 Pro, M1 Max, M1 Ultra) and would like to set it up for data science and machine learning.

**This repo:** Helps you install various software tools such as Homebrew and Miniforge3 to use to install various data science and machine learning tools such as PyTorch. We'll also be getting PyTorch to run on the Apple Silicon GPU for (hopefully) faster computing.

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

## Results

**Last update:** 23 May 2022

Benchmark results were gathered with the notebook [`00_cifar10_tinyvgg.ipynb`](https://github.com/mrdbourke/pytorch-apple-silicon/blob/main/01_cifar10_tinyvgg.ipynb).

Running TinyVGG on CIFAR10 dataset with batch size 32 and image size 32*32:

![results for running PyTorch on Apple M1 Pro with TinyVGG and CIFAR10]("https://raw.githubusercontent.com/mrdbourke/pytorch-apple-silicon/main/results/TinyVGG_cifar10_benchmark_with_batch_size_32_image_size_32.png")

Running TinyVGG on CIFAR10 dataset with batch size 32 and image size 224*224:

![]

## How to setup a PyTorch environment on Apple Silicon using Miniforge (longer version)

If you're new to creating environments, using an Apple Silicon Mac (M1, M1 Pro, M1 Max, M1 Ultra) machine and would like to get started running PyTorch and other data science libraries, follow the below steps.

> **Note:** You're going to see the term "package manager" a lot below. Think of it like this: a **package manager** is a piece of software that helps you install other pieces (packages) of software.

### Installing package managers (Homebrew and Miniforge)

1. Download and install Homebrew from https://brew.sh. Homebrew is a package manager that sets up a lot of useful things on your machine, including Command Line Tools for Xcode, you'll need this to run things like `git`. The command to install Homebrew will look something like:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It will explain what it's doing and what you need to do as you go.

2. [Download the most compatible version of Miniforge](https://github.com/conda-forge/miniforge#download) (minimal installer for Conda specific to conda-forge, Conda is another package manager and conda-forge is a Conda channel) from GitHub.

If you're using an M1 variant Mac, it's "[Miniforge3-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)" <- click for direct download. 

Clicking the link above will download a shell file called `Miniforge3-MacOSX-arm64.sh` to your `Downloads` folder (unless otherwise specified). 

3. Open Terminal.

4. We've now got a shell file capable of installing Miniforge, but to do so we'll have to modify it's permissions to [make it executable](https://askubuntu.com/tags/chmod/info).

To do so, we'll run the command `chmod -x FILE_NAME` which stands for "change mode of FILE_NAME to -executable".

We'll then execute (run) the program using `sh`.

```bash
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
```

5. This should install Miniforge3 into your home directory (`~/` stands for "Home" on Mac).

To check this, we can try to activate the `(base)` environment, we can do so using the `source` command.

```bash
source ~/miniforge3/bin/activate
```

If it worked, you should see something like the following in your terminal window.

```bash
(base) daniel@Daniels-MBP ~ %
```

6. We've just installed some new software and for it to fully work, we'll need to **restart terminal**. 

### Creating a PyTorch environment

Now we've got the package managers we need, it's time to install PyTorch.

Let's setup a folder called `pytorch-test` (you can call this anything you want) and install everything in there to make sure it's working.

> **Note:** An **environment** is like a virtual room on your computer. For example, you use the kitchen in your house for cooking because it's got all the tools you need. It would be strange to have an oven in your bedroom. The same thing on your computer. If you're going to be working on specific software, you'll want it all in one place and not scattered everywhere else. 

7. Make a directory called `pytorch-test`. This is the directory we're going to be storing our environment. And inside the environment will be the software tools we need to run PyTorch, especially PyTorch on the Apple Silicon GPU.

We can do so with the `mkdir` command which stands for "make directory".

```bash
mkdir pytorch-test
```

8. Change into `pytorch-test`. For the rest of the commands we'll be running them inside the directory `pytorch-test` so we need to change into it.

We can do this with the `cd` command which stands for "change directory".

```bash
cd pytorch-test
```

9. Now we're inside the `pyorch-test` directory, let's create a new Conda environment using the `conda` command (this command was installed when we installed Miniforge above).

We do so using `conda create --prefix ./env` which stands for "conda create an environment with the name `file/path/to/this/folder/env`". The `.` stands for "everything before".

For example, if I didn't use the `./env`, my filepath looks like: `/Users/daniel/pytorch-test/env`

```bash
conda create --prefix ./env
```

10. Activate the environment. If `conda` created the environment correctly, you should be able to activate it using `conda activate path/to/environment`.

Short version: 

```bash
conda activate ./env
```

Long version:

```bash
conda activate /Users/daniel/pytorch-test/env
```

> **Note:** It's important to activate your environment every time you'd like to work on projects that use the software you install into that environment. For example, you might have one environment for every different project you work on. And all of the different tools for that specific project are stored in its specific environment.

If activating your environment went correctly, your terminal window prompt should look something like: 

```bash
(/Users/daniel/pytorch-test/env) daniel@Daniels-MBP pytorch-test %
```

11. Now we've got a Conda environment setup, it's time to install the software we need.

Let's start by installing the nightly version of PyTorch for Mac from the [PyTorch install page](https://pytorch.org/get-started/locally/).

> **Note:** 
> As of May 21 2022, accelerated PyTorch for Mac (PyTorch using the Apple Silicon GPU) is still in beta, so expect some rough edges.

```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

The above stands for "hey pip install all of the PyTorch and torch dependencies from the nightly PyTorch channel".

If it worked, you should see a bunch of stuff being downloaded and installed for you. 

12. Install common data science packages. 

If you'd like to work on other various data science and machine learning projects, you're likely going to need Jupyter Notebooks, pandas for data manipulation, NumPy for numeric computing, matplotlib for plotting and Scikit-Learn for traditional machine learning algorithms and processing functions.

To install those in the current environment run:

```bash
conda install jupyter pandas numpy matplotlib scikit-learn tqdm
```

13. Test it out. To see if everything worked, try starting a Jupyter Notebook and importing the installed packages.

```bash
# Start a Jupyter notebook
jupyter notebook
```

Once the notebook is started, in the first cell:

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

14. To run data/models on an Apple Silicon GPU, use the PyTorch device name `"mps"` with `.to("mps")`. MPS stands for *Metal Performance Shaders*, [Metal is Apple's GPU framework](https://developer.apple.com/metal/). 

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

15. To see if it really worked, try running one of the notebooks above end to end!