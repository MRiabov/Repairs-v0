# Overview 
Neural network model(s) to solve [RepairsComponents-v0](github.com/MRiabov/RepairsComponents-v0)

### Algorithms include:
- SAC - Soft
- BRONet - Bigger, Regularized, Optimistic: scaling for compute and sample efficient continuous control (NeurlIPS 2025) https://arxiv.org/html/2405.16158v1 ***Work in progress***
- SimbaV2 - Hyperspherical Normalization for Scalable Deep Reinforcement Learning (ICML'25 (spotlight)) https://arxiv.org/abs/2502.15280 ***Work in progress***

### Features:
- A configured [RepairsComponents-v0](github.com/MRiabov/RepairsComponents-v0) out-of-the-box.
- A highly optimal (singleton) buffer for one-use observations (e.g. graphs) for the beginning and finish of simulation,
- Graph encoders for mechanical and electronics simulation,
- Vision (CNN) encoders for video observations.

### Usage:
Follow every command from `setup.sh` from [RepairsComponents-v0](github.com/MRiabov/RepairsComponents-v0):
```sh
sudo apt install libgl1-mesa-dev libsparsehash-dev -y

pip install uv
cd RepairsComponents-v0/
uv venv
source .venv/bin/activate
pip install uv
uv pip install -r /workspace/RepairsComponents-v0/combined_req.txt -U # faster install of torchsparse
uv pip install torch==2.5.1 torchvision setuptools git+https://github.com/gumyr/build123d
uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

uv pip install numpy==1.26.4 --no-deps # note: I got this working by using not uv but standard pip.
uv pip install -e /workspace/RepairsComponents-v0/.  --no-deps
uv pip install git+https://github.com/mit-han-lab/torchsparse --no-build-isolation

```

Note: Under active development. Paper is published very soon.
