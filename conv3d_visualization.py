import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_unet_like_architecture():
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    
    # Initial dimensions
    input_channels = 3
    input_size = 32
    kernel_size = 3
    pool_size = 2
    
    # Channel progression (like U-Net)
    channel_progression = [input_channels, 64, 128, 256]
    
    # Create sample data
    input_data = np.random.rand(input_channels, input_size, input_size)
    
    # Visualize each stage
    for stage in range(len(channel_progression)):
        # Calculate current size
        current_size = input_size // (2**stage)
        current_channels = channel_progression[stage]
        
        # Create sample data for this stage
        stage_data = np.random.rand(current_channels, current_size, current_size)
        
        # Visualize first few channels of this stage
        num_channels_to_show = min(3, current_channels)
        for i in range(num_channels_to_show):
            ax = fig.add_subplot(3, 3, stage*3 + i + 1)
            im = ax.imshow(stage_data[i], cmap='viridis')
            ax.set_title(f'Stage {stage+1}\nChannels: {current_channels}\nSize: {current_size}x{current_size}')
            plt.colorbar(im, ax=ax)
    
    # Add text explanation
    explanation = """
    U-Net-like Architecture:
    
    Stage 1: (3, 32, 32)
    - Input image (e.g., RGB)
    
    Stage 2: (64, 16, 16)
    - After first conv+pool block
    - Channels increased to 64
    - Spatial dimensions halved
    
    Stage 3: (128, 8, 8)
    - After second conv+pool block
    - Channels increased to 128
    - Spatial dimensions halved again
    
    Stage 4: (256, 4, 4)
    - After third conv+pool block
    - Channels increased to 256
    - Spatial dimensions halved again
    
    Note: Each stage typically:
    1. Doubles the number of channels
    2. Halves the spatial dimensions
    """
    
    plt.figtext(0.02, 0.02, explanation, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('unet_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_unet_like_architecture() 