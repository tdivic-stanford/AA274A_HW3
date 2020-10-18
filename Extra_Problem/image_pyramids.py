#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

def half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    return image[::2, ::2, :]
    ########## Code ends here ##########


def blur_half_downscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        downscaled_image: A half-downscaled version of image.
    """
    ########## Code starts here ##########
    # call the half_downscale function on a blurred image
    return half_downscale(cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.7))
    ########## Code ends here ##########


def two_upscale(image):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
    
    Returns
        upscaled_image: A 2x-upscaled version of image.
    """
    ########## Code starts here ##########
    raise NotImplementedError("Implement me!")
    ########## Code ends here ##########


def bilinterp_upscale(image, scale):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing an m x n image (with c channels).
        scale: How much larger to make the image

    Returns
        upscaled_image: A scale-times upscaled version of image.
    """
    m, n, c = image.shape

    f = (1./scale) * np.convolve(np.ones((scale, )), np.ones((scale, )))
    f = np.expand_dims(f, axis=0) # Making it (1, (2*scale)-1)-shaped
    filt = f.T * f

    ########## Code starts here ##########
    raise NotImplementedError("Implement me!")
    ########## Code ends here ##########

def show_save_img(filename, image):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    ax.imshow(image, interpolation='none')
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)

def main():
    # OpenCV actually uses a BGR color channel layout,
    # Matplotlib uses an RGB color channel layout, so we're flipping the 
    # channels here so that plotting matches what we expect for colors.
    test_card = cv2.imread('test_card.png')[..., ::-1].astype(float)
    favicon = cv2.imread('favicon-16x16.png')[..., ::-1].astype(float)
    test_card /= test_card.max()
    favicon /= favicon.max()

    # Note that if you call matplotlib's imshow function to visualize images,
    # be sure to pass in interpolation='none' so that the image you see
    # matches exactly what's in the data array you pass in.
    
    ########## Code starts here ##########
    test_card_downscaled = half_downscale(half_downscale(half_downscale(test_card)))
    show_save_img("test_card_downscaled.png", test_card_downscaled)

    blurred_test_card_downscaled = blur_half_downscale(blur_half_downscale(blur_half_downscale(test_card)))
    show_save_img("blurred_test_card_downscaled.png", blurred_test_card_downscaled)

    ########## Code ends here ##########


if __name__ == '__main__':
    main()
