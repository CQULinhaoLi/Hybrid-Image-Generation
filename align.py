import cv2
import numpy as np
import matplotlib.pyplot as plt

def select_points(image, title="Select points"):
    """
    Select points manually on an image.
    :param image: Input image (H, W, 3)
    :param title: Title of the plot
    :return: List of selected points
    """
    plt.imshow(image)
    plt.title(title)
    points = plt.ginput(3)  # Select 3 points
    plt.close()
    return np.array(points, dtype=np.float32)

def align_images(image1, image2):
    """
    Align image2 to image1 using affine transformation.
    :param image1: Reference image (H, W, 3)
    :param image2: Target image to be aligned (H, W, 3)
    :return: Aligned version of image2
    """
    # Select corresponding points from both images
    print("Select 3 points on the first image.")
    points1 = select_points(image1)
    print("Select the same 3 points on the second image.")
    points2 = select_points(image2)

    # Compute the affine transformation matrix
    M = cv2.getAffineTransform(points2, points1)

    # Apply the affine transformation
    aligned_image = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))

    return aligned_image
if __name__ == "__main__":
    # Load images
    image1 = plt.imread('images/tiger.jpg') / 255.0
    image2 = plt.imread('images/lion.jpg') / 255.0

    # Align image2 to image1
    aligned_image2 = align_images(image1, image2)

    # Display aligned image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(aligned_image2)
    plt.title("Aligned Image")
    plt.axis("off")

    plt.show()
