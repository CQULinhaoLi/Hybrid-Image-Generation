import numpy as np
from PIL import Image

def gaussian_kernel(kernel_size, sigma):
    """Generate a Gaussian kernel."""
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def gaussian_blur(image, kernel):
    """Apply Gaussian blur using the given kernel."""
    h, w, c = image.shape
    pad = kernel.shape[0] // 2
    padded_image = np.pad(image, [(pad, pad), (pad, pad), (0, 0)], mode='reflect')
    blurred = np.zeros_like(image, dtype=np.float32)

    for x in range(h):
        for y in range(w):
            for ch in range(c):
                blurred[x, y, ch] = np.sum(
                    kernel * padded_image[x:x + kernel.shape[0], y:y + kernel.shape[1], ch]
                )
    return blurred

def downsample(image):
    """Downsample the image by a factor of 2."""
    return image[::2, ::2]

def upsample(image):
    """Upsample the image by a factor of 2."""
    h, w, c = image.shape
    upsampled = np.zeros((h * 2, w * 2, c), dtype=image.dtype)
    upsampled[::2, ::2] = image

    # Fill the gaps using average of neighbors
    for i in range(h * 2):
        for j in range(w * 2):
            if i % 2 == 1 or j % 2 == 1:
                neighbors = []
                if i > 0:
                    neighbors.append(upsampled[i - 1, j])
                if i < h * 2 - 1:
                    neighbors.append(upsampled[i + 1, j])
                if j > 0:
                    neighbors.append(upsampled[i, j - 1])
                if j < w * 2 - 1:
                    neighbors.append(upsampled[i, j + 1])
                upsampled[i, j] = np.mean(neighbors, axis=0)
    return upsampled

def gaussian_pyramid(image, levels, kernel):
    """Build Gaussian Pyramid."""
    pyramid = [image]
    for _ in range(levels):
        image = gaussian_blur(image, kernel)
        image = downsample(image)
        pyramid.append(image)
    return pyramid

def laplacian_pyramid(gaussian_pyramid):
    """Build Laplacian Pyramid."""
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        upsampled = upsample(gaussian_pyramid[i + 1])
        upsampled = upsampled[:gaussian_pyramid[i].shape[0], :gaussian_pyramid[i].shape[1]]
        laplacian = gaussian_pyramid[i] - upsampled
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid

def reconstruct_image(laplacian_pyramid):
    """Reconstruct image from Laplacian Pyramid."""
    image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        upsampled = upsample(image)
        upsampled = upsampled[:laplacian_pyramid[i].shape[0], :laplacian_pyramid[i].shape[1]]
        image = upsampled + laplacian_pyramid[i]
    return image

def pyramid_blending(img1, img2, mask, levels, kernel):
    """Perform Pyramid Blending."""
    gp1 = gaussian_pyramid(img1, levels, kernel)
    gp2 = gaussian_pyramid(img2, levels, kernel)
    gpm = gaussian_pyramid(mask, levels, kernel)

    lp1 = laplacian_pyramid(gp1)
    lp2 = laplacian_pyramid(gp2)

    blended_pyramid = []
    for l1, l2, gm in zip(lp1, lp2, gpm):
        blended = l1 * gm + l2 * (1 - gm)
        blended_pyramid.append(blended)

    return reconstruct_image(blended_pyramid)

def load_image(path):
    """Load image using PIL."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def save_image(image, path):
    """Save image using PIL."""
    img = Image.fromarray((image * 255).astype(np.uint8))
    img.save(path)

def create_horizontal_gradient_mask(h, w):
    """Create horizontal gradient mask."""
    mask = np.zeros((h, w), dtype=np.float32)
    for i in range(w):
        mask[:, i] = i / (w - 1)
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Expand to 3 channels
    return mask

# Main script
if __name__ == "__main__":
    # Load images
    img1 = load_image(r"blending_images\\Pyramid_Blending_img\\apple.png")
    img2 = load_image(r"blending_images\\Pyramid_Blending_img\\orange.png")

    # Create horizontal gradient mask
    h, w, _ = img1.shape
    mask = create_horizontal_gradient_mask(h, w)

    # Define Gaussian kernel
    kernel = gaussian_kernel(kernel_size=5, sigma=1.0)

    # Perform blending
    blended_image = pyramid_blending(img1, img2, mask, levels=5, kernel=kernel)

    # Save result
    save_image(blended_image, r"blending_images\\Pyramid_Blending_img\\blended_image.jpg")