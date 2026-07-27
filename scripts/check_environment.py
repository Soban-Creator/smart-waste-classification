"""
Environment Check Script
------------------------
This script checks whether all important libraries are installed correctly.
"""

import sys

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image


def main():
    print("=" * 50)
    print(" SMART WASTE CLASSIFICATION")
    print(" Environment Check")
    print("=" * 50)

    print(f"\nPython Version: {sys.version}")
    print(f"NumPy Version: {np.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Scikit-Learn Version: {sklearn.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchVision Version: {torchvision.__version__}")
    print(f"Pillow Version: {Image.__version__}")

    print("\nChecking Hardware...")

    if torch.cuda.is_available():
        print("GPU Available")
        print(torch.cuda.get_device_name(0))
    else:
        print("GPU Not Available")
        print("Using CPU")

    print("\nEverything looks good!")
    print("=" * 50)


if __name__ == "__main__":
    main()