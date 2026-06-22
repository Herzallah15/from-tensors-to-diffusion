# From Tensors to Diffusion

A hands-on, mathematically grounded course in deep learning with **PyTorch** - starting from the elementary building blocks of a neural network and ending at modern **deep generative models**. Every topic is developed from its mathematical definition and paired with runnable code, so the notebooks read both as a reference and as a guided curriculum.

The material is organized as a single progression:

> **tensors → layers → training → architectures → latent variable models → diffusion**

---

## Curriculum

The notebooks are numbered and meant to be followed in order. Each part builds on the previous one.

### Part I - Foundations (`01-foundations/`)
The components every network is assembled from, each treated as a parameterized map and its
associated tensor shapes.

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01-parametric-layers.ipynb` | Linear, convolutional and recurrent layers; input/output tensor shapes and when each is used. |
| 02 | `02-nonlinear-activations.ipynb` | Nonlinearities (ReLU, Sigmoid, GELU, …): definitions, properties, and how the choice affects training. |
| 03 | `03-loss-functions.ipynb` | Objective functions for regression and classification, with their mathematical form and use cases. |
| 04 | `04-gradient-based-optimization.ipynb` | Optimizers (SGD, ASGD, Adam, …): update rules, trade-offs, and convergence behavior. |
| 05 | `05-data-transforms-and-dataloaders.ipynb` | Transforms, custom/composed transforms, and the `Dataset` / `DataLoader` pipeline. |

### Part II - Architectures (`02-architectures/`)
The foundations applied end-to-end to three canonical learning problems.

| # | Notebook | Topic |
|---|----------|-------|
| 06 | `06-feedforward-networks-and-inverse-modeling.ipynb` | Feedforward networks for an **inverse problem**: recovering the parameters of a lattice-QCD-inspired correlator. |
| 07 | `07-convolutional-networks-for-image-classification.ipynb` | CNNs for image classification on **Fashion-MNIST**. |
| 08 | `08-recurrent-networks-for-sequence-classification.ipynb` | RNNs for text classification on the **AG News** dataset. |

### Part III - Generative Models (`03-generative-models/`)
From representation learning to sampling new data.

| # | Notebook | Topic |
|---|----------|-------|
| 09 | `09-latent-variable-models-and-autoencoders.ipynb` | Latent variable models and the autoencoder as a foundation for generative modeling. |
| 10 | `10-variational-autoencoders.ipynb` | Variational autoencoders: the latent manifold, the reparameterization trick, and the ELBO. |
| 11 | `11-diffusion-models.ipynb` | Forward/reverse diffusion and the HuggingFace **Diffusers** library. |
| 12 | `12-diffusion-models-data-pipeline.ipynb` | Building and loading a custom image dataset to train a diffusion model. |

---

## Prerequisites

- Comfort with Python and basic linear algebra, calculus, and probability.
- Python **3.10+**.
- A GPU is recommended for Parts II and III (especially the diffusion notebooks) but not required.

## Setup

```bash
# clone
git clone https://github.com/Herzallah15/from-tensors-to-diffusion.git
cd from-tensors-to-diffusion

# (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install dependencies
pip install torch torchvision \
            diffusers transformers accelerate \
            numpy pandas scikit-learn matplotlib pillow tqdm \
            jupyter
```

> Install the `torch` / `torchvision` build that matches your CUDA version by following the
> selector at <https://pytorch.org/get-started/locally/>.

## Running the notebooks

```bash
jupyter lab        # or: jupyter notebook
```

Open the notebooks in numerical order, starting with `01-foundations/`.

---

## Repository layout

```
from-tensors-to-diffusion/
├── README.md
├── 01-foundations/
│   ├── 01-parametric-layers.ipynb
│   ├── 02-nonlinear-activations.ipynb
│   ├── 03-loss-functions.ipynb
│   ├── 04-gradient-based-optimization.ipynb
│   └── 05-data-transforms-and-dataloaders.ipynb
├── 02-architectures/
│   ├── 06-feedforward-networks-and-inverse-modeling.ipynb
│   ├── 07-convolutional-networks-for-image-classification.ipynb
│   └── 08-recurrent-networks-for-sequence-classification.ipynb
└── 03-generative-models/
    ├── 09-latent-variable-models-and-autoencoders.ipynb
    ├── 10-variational-autoencoders.ipynb
    ├── 11-diffusion-models.ipynb
    └── 12-diffusion-models-data-pipeline.ipynb
```

## License

Released under the MIT License.
