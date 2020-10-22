#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt

def template_match(template, image,
                   num_upscales=2, num_downscales=3,
                   detection_threshold=0.93):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        num_upscales: How many times to 2x-upscale image with Gaussian blur before template matching over it.
        num_downscales: How many times to 0.5x-downscale image with Gaussian blur before template matching over it.
        detection_threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    # create empty matches array
    matches = []

    # first get the matches for the original image
    new_matches = problem4_template_matching(template, image, threshold=detection_threshold)
    matches.extend(new_matches)

    # then loop through the upscales
    upscaled_image = image
    for i in range(num_upscales):
        # upscale the last upscaled image and save it as the new upscaled image
        upscaled_image = cv2.pyrUp(upscaled_image)

        # template match on this new image
        new_matches = problem4_template_matching(template, upscaled_image, threshold=detection_threshold)

        # scale the new matches back down to original size
        sf = (i + 1) * 2 # scaling factor
        new_matches = [(y/sf, x/sf, h/sf, w/sf) for y, x, h, w in new_matches]

        # append the new matches to our list of matches
        matches.extend(new_matches)

    # finally, loop through the downscales
    downscaled_image = image
    for i in range(num_downscales):
        # downscale the image
        downscaled_image = cv2.pyrDown(downscaled_image)

        # template match
        new_matches = problem4_template_matching(template, downscaled_image, threshold=detection_threshold)

        # scale the new matches back up to original size
        sf = (i + 1) * 2  # scaling factor
        new_matches = [(y * sf, x * sf, h * sf, w * sf) for y, x, h, w in new_matches]

        # append to our list
        matches.extend(new_matches)

    return matches
    ########## Code ends here ##########


def problem4_template_matching(template, image, threshold=0.999):
    """
    Input
        template: A (k, ell, c)-shaped ndarray containing the k x ell template (with c channels).
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        threshold: Minimum normalized cross-correlation value to be considered a match.

    Returns
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.
    """
    ########## Code starts here ##########
    results = cv2.matchTemplate(image, template, method=cv2.TM_CCORR_NORMED)

    # find all points in the resulting grid that exceed our matching threshold
    match_corners = np.argwhere(results > threshold)

    # for all the match corners, create the bounding box and add it to our matches
    matches = []
    width = np.shape(template)[1]
    height = np.shape(template)[0]
    for match_corner in match_corners:
        matches.append((match_corner[0], match_corner[1], height, width))

    return matches
    ########## Code ends here ##########


def create_and_save_detection_image(image, matches, filename="image_detections.png"):
    """
    Input
        image: An (m, n, c)-shaped ndarray containing the m x n image (with c channels).
        matches: A list of (top-left y, top-left x, bounding box height, bounding box width) tuples for each match's bounding box.

    Returns
        None, this function will save the detection image in the current folder.
    """
    det_img = image.copy()
    for (y, x, bbox_h, bbox_w) in matches:
        cv2.rectangle(det_img, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

    cv2.imwrite(filename, det_img)


def main():
    template = cv2.imread('messi_face.jpg')
    image = cv2.imread('messipyr.jpg')

    matches = template_match(template, image)
    create_and_save_detection_image(image, matches)

    template = cv2.imread('stop_signs/stop_template.jpg').astype(np.float32)
    for i in range(1, 6):
        image = cv2.imread('stop_signs/stop%d.jpg' % i).astype(np.float32)
        matches = template_match(template, image, detection_threshold=0.87)
        create_and_save_detection_image(image, matches, 'stop_signs/stop%d_detection.png' % i)


if __name__ == '__main__':
    main()
