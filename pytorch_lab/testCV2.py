from kornia.utils import image_to_tensor
from kornia.utils import tensor_to_image
import numpy as np
import torch
import cv2
from PIL import Image

def test1():
    x_np = np.ones((5, 5))
    print(x_np)

    x = image_to_tensor(x_np, keepdim=False)
    print("--------\n", x, "--------\n", x.shape) ## [Batch, Channel, Row, Col]

def test2():
    x = torch.randn((1, 3, 5, 5))
    print("\n========\n", x, "\n========\n", x.shape) 

    x_np = tensor_to_image(x)
    print("\n--------\n", x_np, "\n--------\n", x_np.shape)  ## [Row, Col, Channel]


if __name__ == "__main__":
    test2()