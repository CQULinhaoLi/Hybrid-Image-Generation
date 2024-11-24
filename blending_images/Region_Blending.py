import numpy as np
from PIL import Image
from Pyramid_Blending import*
# 已有的函数保持不变（如 Gaussian Kernel, Gaussian Blur, Downsample, Upsample 等）

def create_elliptical_mask(h, w, center, axes, angle=0):
    """
    Create an elliptical mask for Region Blending.
    :param h: Height of the mask
    :param w: Width of the mask
    :param center: Center of the ellipse (x, y)
    :param axes: Lengths of the ellipse axes (x_radius, y_radius)
    :param angle: Rotation angle of the ellipse (in degrees)
    :return: A 2D elliptical mask normalized to [0, 1]
    """
    mask = np.zeros((h, w), dtype=np.float32)  # Initialize a 2D mask
    for y in range(h):
        for x in range(w):
            # Transform the coordinates to account for rotation
            x_rot = (x - center[0]) * np.cos(np.radians(angle)) + (y - center[1]) * np.sin(np.radians(angle))
            y_rot = -(x - center[0]) * np.sin(np.radians(angle)) + (y - center[1]) * np.cos(np.radians(angle))
            # Check if the point lies within the ellipse
            if (x_rot / axes[0])**2 + (y_rot / axes[1])**2 <= 1:
                mask[y, x] = 1.0
    return np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Convert to 3-channel mask

def region_blending(img1, img2, mask, levels, kernel):
    """
    Perform Region Blending using Laplacian Pyramid.
    :param img1: First image
    :param img2: Second image
    :param mask: 3-channel mask defining blending region
    :param levels: Number of pyramid levels
    :param kernel: Gaussian kernel for pyramid construction
    :return: Blended image
    """
    # Build pyramids for the two images and the mask
    gp1 = gaussian_pyramid(img1, levels, kernel)
    gp2 = gaussian_pyramid(img2, levels, kernel)
    gpm = gaussian_pyramid(mask, levels, kernel)

    lp1 = laplacian_pyramid(gp1)
    lp2 = laplacian_pyramid(gp2)

    # Blend Laplacian pyramids
    blended_pyramid = []
    for l1, l2, gm in zip(lp1, lp2, gpm):
        blended = l1 * gm + l2 * (1 - gm)
        blended_pyramid.append(blended)

    # Reconstruct the final blended image
    return reconstruct_image(blended_pyramid)

def resize_image(image, target_size):
    """Resize image to the target size (height, width)."""
    return np.array(Image.fromarray((image * 255).astype(np.uint8)).resize(target_size[::-1])).astype(np.float32) / 255.0

# Main script for Region Blending
if __name__ == "__main__":
    # Load two images
    img1 = load_image(r"D:/Desktop/CV/CVprojectCode/Region_Blending_img/eye_2.jpg")
    img2 = load_image(r"D:/Desktop/CV/CVprojectCode/Region_Blending_img/hand_2.jpg")

    # Ensure images have the same size
    target_size = (min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1]))
    img1 = resize_image(img1, target_size)
    img2 = resize_image(img2, target_size)

    # Define mask parameters
    h, w, _ = img1.shape
    center = (w // 2, h // 2)  # Center of the ellipse
    axes = (w // 4, h // 4)    # Axes of the ellipse
    angle = 0                  # Angle of the ellipse

    # Create elliptical mask
    mask = create_elliptical_mask(h, w, center, axes, angle)

    # Define Gaussian kernel
    kernel = gaussian_kernel(kernel_size=5, sigma=1.0)

    # Perform region blending
    blended_image = region_blending(img1, img2, mask, levels=5, kernel=kernel)

    # Save the blended image
    save_image(blended_image, r"D:/Desktop/CV/CVprojectCode/Region_Blending_img/region_blended_image.jpg")
    print("Region blended image saved at: D:/Desktop/CV/CVprojectCode/Region_Blending_img/region_blended_image.jpg")

