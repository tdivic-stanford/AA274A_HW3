#!/usr/bin/env python

import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


def corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the correlation of the filter with the image.
    """
    ########## Code starts here ##########
    # get size info about F
    F_shape = np.shape(F)

    # created a padded I
    num_padded_cols = int(F_shape[1] / 2)
    num_padded_rows = int(F_shape[0] / 2)

    # add row padding
    I_padded = np.pad(I, ((num_padded_rows, num_padded_rows), (num_padded_cols, num_padded_cols), (0,0)), 'constant')

    # create the filter vector for our dot product
    f = F.flatten()

    # create size of G so we don't need to extract shape for every loop
    G_rows = np.shape(I)[0]
    G_cols = np.shape(I)[1]
    G = np.zeros((G_rows, G_cols))

    # for each i and j pair, create a t_ij vector and calculate G_ij
    for i in range(1, G_rows + 1):  # keeping this 1-indexed per the pset
        for j in range(1, G_cols + 1):
            # extract the current I_padded sub-matrix
            curr_I = I_padded[(i - 1):(i - 1 + F_shape[0]), (j - 1):(j - 1 + F_shape[1]), 0:F_shape[2]]

            # flatten into the t_ij vector
            t_ij = curr_I.flatten()
            # compute current G element as dot product of f and t_ij
            G[i-1,j-1] = np.dot(f, t_ij)

    return G

    ########## Code ends here ##########


def norm_cross_corr(F, I):
    """
    Input
        F: A (k, ell, c)-shaped ndarray containing the k x ell filter (with c channels).
        I: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).

    Returns
        G: An (m, n)-shaped ndarray containing the normalized cross-correlation of the filter with the image.
    """
    ########## Code starts here ##########
    # get size info about F
    F_shape = np.shape(F)

    # created a padded I
    num_padded_cols = int(F_shape[1] / 2)
    num_padded_rows = int(F_shape[0] / 2)

    # add row padding
    I_padded = np.pad(I, ((num_padded_rows, num_padded_rows), (num_padded_cols, num_padded_cols), (0, 0)), 'constant')

    # create the filter vector for our dot product
    f = F.flatten()

    # create size of G so we don't need to extract shape for every loop
    G_rows = np.shape(I)[0]
    G_cols = np.shape(I)[1]
    G = np.zeros((G_rows, G_cols))

    # for each i and j pair, create a t_ij vector and calculate G_ij
    for i in range(1, G_rows + 1):  # keeping this 1-indexed per the pset
        for j in range(1, G_cols + 1):
            # extract the current I_padded sub-matrix
            curr_I = I_padded[(i - 1):(i - 1 + F_shape[0]), (j - 1):(j - 1 + F_shape[1]), 0:F_shape[2]]

            # flatten into the t_ij vector
            t_ij = curr_I.flatten()

            # compute current G element as dot product of f and t_ij, divided by the norms
            G[i - 1, j - 1] = np.dot(f, t_ij) / (np.linalg.norm(f) * np.linalg.norm(t_ij))

    return G
    ########## Code ends here ##########


def show_save_corr_img(filename, image, template):
    # Not super simple, because need to normalize image scale properly.
    fig, ax = plt.subplots()
    cropped_img = image[:-template.shape[0], :-template.shape[1]]
    im = ax.imshow(image, interpolation='none', vmin=cropped_img.min())
    fig.colorbar(im)
    fig.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def main():
    test_card = cv2.imread('test_card.png').astype(np.float32)

    filt1 = np.zeros((3, 3, 1))
    filt1[1, 1] = 1

    filt2 = np.zeros((3, 200, 1))
    filt2[1, -1] = 1

    filt3 = np.zeros((3, 3, 1))
    filt3[:, 0] = -1
    filt3[:, 2] = 1

    filt4 = (1./273.)*np.array([[1, 4, 7, 4, 1],
                              [4, 16, 26, 16, 4],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4, 7, 4, 1]])
    filt4 = np.expand_dims(filt4, -1)

    grayscale_filters = [filt1, filt2, filt3, filt4]

    color_filters = list()
    for filt in grayscale_filters:
        # Making color filters by replicating the existing
        # filter per color channel.
        color_filters.append(np.concatenate([filt, filt, filt], axis=-1))

    for idx, filt in enumerate(color_filters):
        start = time.time()
        corr_img = corr(filt, test_card)
        stop = time.time()
        print 'Correlation function runtime:', stop - start, 's'
        show_save_corr_img("corr_img_filt%d.png" % idx, corr_img, filt)


if __name__ == "__main__":
    main()
