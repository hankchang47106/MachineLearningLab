import cv2
import numpy as np
import torch
import torchvision
import kornia
import matplotlib.pyplot as plt

def load_cv_image(fn: str, isShow=True) -> np.ndarray:
    """
    Obj: load a cv image from fn
    """
    cvImage = cv2.imread(fn, -1)
    if isShow:
        cv2.imshow(fn, cvImage)
        cv2.waitKey(0)
    return cvImage

def main():
    fn = "./Image/lenna.png"
    img_bgr: np.ndarray = load_cv_image(fn)
    x_bgr: torch.tensor = kornia.image_to_tensor(img_bgr)
    print(x_bgr)
    x_rgb = kornia.color.bgr_to_rgb(x_bgr)
    print(x_rgb.shape)

if __name__ == "__main__":
    main()
    