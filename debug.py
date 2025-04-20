import torch
import numpy as np
from sklearn import __version__ as sk_version
import matplotlib
import tensorboard

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"NumPy版本: {np.__version__}")
print(f"scikit-learn版本: {sk_version}")
print(f"Matplotlib版本: {matplotlib.__version__}")
print(f"TensorBoard版本: {tensorboard.__version__}")
