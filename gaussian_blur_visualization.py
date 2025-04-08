import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def create_gaussian_kernel(sigma, size):
    """Create a 2D Gaussian kernel."""
    x = np.linspace(-(size // 2), size // 2, size)
    y = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(x, y)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()  # Normalize

def visualize_gaussian_blur():
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4)  # Changed to 2x4 grid
    
    # Create sample image (a simple pattern)
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    sample_image = np.sin(X) * np.cos(Y)
    
    # Plot original image
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(sample_image, cmap='viridis')
    ax1.set_title('Original Image')
    plt.colorbar(im1, ax=ax1)
    
    # Create and plot Gaussian kernels of different sizes
    sigmas = [1, 2, 4]
    kernel_sizes = [5, 11, 21]
    
    for idx, (sigma, size) in enumerate(zip(sigmas, kernel_sizes)):
        kernel = create_gaussian_kernel(sigma, size)
        
        # Plot kernel
        ax = fig.add_subplot(gs[0, idx + 1])
        im = ax.imshow(kernel, cmap='viridis')
        ax.set_title(f'Gaussian Kernel\nσ={sigma}, size={size}x{size}')
        plt.colorbar(im, ax=ax)
        
        # Apply blur and plot result
        blurred = gaussian_filter(sample_image, sigma=sigma)
        ax_blur = fig.add_subplot(gs[1, idx])
        im_blur = ax_blur.imshow(blurred, cmap='viridis')
        ax_blur.set_title(f'Blurred Image\nσ={sigma}')
        plt.colorbar(im_blur, ax=ax_blur)
    
    # Add computational complexity plot
    ax_complexity = fig.add_subplot(gs[1, 3])  # Changed to last column
    sizes = np.arange(3, 51, 2)
    operations = sizes**2  # Number of operations per pixel
    ax_complexity.plot(sizes, operations, 'r-', label='Operations per pixel')
    ax_complexity.set_xlabel('Kernel Size')
    ax_complexity.set_ylabel('Number of Operations')
    ax_complexity.set_title('Computational Complexity')
    ax_complexity.grid(True)
    ax_complexity.legend()
    
    plt.tight_layout()
    plt.savefig('gaussian_blur_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_gaussian_blur() 