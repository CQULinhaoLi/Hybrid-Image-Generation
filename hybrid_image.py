import matplotlib.pyplot as plt
from pyramids import *
from align import *
import matplotlib.gridspec as gridspec

from skimage.transform import resize



def main_gaussian_laplacian_pyramids(image, kernel, levels):
    """
    A function to build the Gaussian and Laplacian pyramids of an image
    :param image: A grayscale or 3 channels image, a numpy array of floats within [0, 1] of shape (N, M) or (N, M, 3)
    :param kernel: The Gaussian kernel used to build pyramids
    :param levels: The desired levels in the pyramids
    """

    image = convert_image_to_floats(image)

    # Building the Gaussian and Laplacian pyramids
    gauss_pyr = gaussian_pyramid(image, kernel, levels)
    lap_pyr = laplacian_pyramid(image, kernel, levels)

    gauss_plot_num = len(gauss_pyr)
    lap_plot_num = len(lap_pyr)
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, gauss_plot_num, width_ratios=[x for x in range(gauss_plot_num, 0, -1)])  # 设置子图宽度比例
    for i, p in enumerate(gauss_pyr):
        ax = fig.add_subplot(gs[i])
        ax.imshow(p, cmap='gray')
        ax.set_title(f"GaussianPyramid {i + 1}")
        ax.axis('off')  # hide the axis

    plt.tight_layout()
    plt.savefig('OutputImages/dogGaussianPyramids.png')
    plt.show()

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, lap_plot_num, width_ratios=[x for x in range(1, lap_plot_num+1)])  # 设置子图宽度比例
    for i, p in enumerate(lap_pyr):
        ax = fig.add_subplot(gs[i])
        ax.imshow(p, cmap='gray')
        ax.set_title(f"LaplacianPyramid {lap_plot_num - i}")
        ax.axis('off')  # hide the axis

    plt.tight_layout()
    # plt.savefig('OutputImages/tigerLaplacianPyramids.png')
    plt.show()

    # Building and displaying collapsed image
    # collapsed_image = collapse_pyramid(lap_pyr, gauss_pyr)
    # plt.imshow(collapsed_image, cmap='gray')
    # plt.show()
    return gauss_pyr, lap_pyr

def convert_image_to_floats(image):
    """
    A function to convert an image to a numpy array of floats within [0, 1]

    :param image: The image to be converted
    :return: The converted image
    """

    if np.max(image) <= 1.0:
        return image
    else:
        return image / 255.0

def rgb_to_gray(image):
    """
    A function to convert an image from rgb to gray scale
    :param image:  The image to be converted
    :return: The converted image
    """

    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

kernel = smooth_gaussian_kernel(0.4)
# kernel = classical_gaussian_kernel(5, 1)
levels = 5
window_size = 5


image1 = plt.imread('images/dog.png')
image2 = plt.imread('images/tiger.jpg')
# to align the image to improve the result (optional)
# three points of each image should be selected
# image2 = align_images(image1, image2)

# resize the two images to get better result.
image_size = (2**(levels+3), 2**(levels+3))
image1 = resize(image1, image_size)
image2 = resize(image2, image_size)
image3 = rgb_to_gray(image1)

gau1, lap1 = main_gaussian_laplacian_pyramids(image1, kernel, levels)
gau2, lap2 = main_gaussian_laplacian_pyramids(image2, kernel, levels)

# choose pyramids to hybridize
GAU = gau1[1]
LAP1 = lap2[-2]
LAP2 = lap2[-1]
# To ensure pyramid images can be added together.
LAP1 = resize(LAP1, GAU.shape)
LAP2 = resize(LAP2, GAU.shape)

# set different weight for images to be hybridized.
alpha = 0.3
beta1 = 0.2
beta2 = 0.9

hybrid_image = alpha * GAU + beta1 * LAP1 + beta2 * LAP2


hybrid_image = normalize(hybrid_image)
plt.imshow(hybrid_image)
plt.savefig('OutputImages/HybridImages/HI_3.png')
plt.show()
