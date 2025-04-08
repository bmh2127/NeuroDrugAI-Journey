import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_receptive_field():
    # Create figure with two rows
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original input size
    input_size = 32
    
    # Function to calculate receptive field size
    def get_rf_size(kernel_size, layer):
        if layer == 1:
            return kernel_size
        return get_rf_size(kernel_size, layer-1) + (kernel_size - 1)
    
    # Visualize for different kernel sizes
    kernel_sizes = [3, 5, 7]
    colors = ['red', 'blue', 'green']
    
    for idx, (kernel_size, color) in enumerate(zip(kernel_sizes, colors)):
        # First layer
        ax = [ax1, ax4][idx]
        ax.set_xlim(0, input_size)
        ax.set_ylim(0, input_size)
        ax.set_title(f'Input Layer\n(32x32)')
        ax.grid(True)
        
        # Add receptive field for first layer
        rf1 = Rectangle((16-kernel_size/2, 16-kernel_size/2), 
                       kernel_size, kernel_size, 
                       facecolor=color, alpha=0.3)
        ax.add_patch(rf1)
        ax.text(16, 16, f'{kernel_size}x{kernel_size}', 
                ha='center', va='center')
        
        # Second layer
        ax = [ax2, ax5][idx]
        ax.set_xlim(0, input_size)
        ax.set_ylim(0, input_size)
        rf_size = get_rf_size(kernel_size, 2)
        ax.set_title(f'After First Conv Layer\nReceptive Field: {rf_size}x{rf_size}')
        ax.grid(True)
        
        # Add receptive field for second layer
        rf2 = Rectangle((16-rf_size/2, 16-rf_size/2), 
                       rf_size, rf_size, 
                       facecolor=color, alpha=0.3)
        ax.add_patch(rf2)
        ax.text(16, 16, f'{rf_size}x{rf_size}', 
                ha='center', va='center')
        
        # Third layer
        ax = [ax3, ax6][idx]
        ax.set_xlim(0, input_size)
        ax.set_ylim(0, input_size)
        rf_size = get_rf_size(kernel_size, 3)
        ax.set_title(f'After Second Conv Layer\nReceptive Field: {rf_size}x{rf_size}')
        ax.grid(True)
        
        # Add receptive field for third layer
        rf3 = Rectangle((16-rf_size/2, 16-rf_size/2), 
                       rf_size, rf_size, 
                       facecolor=color, alpha=0.3)
        ax.add_patch(rf3)
        ax.text(16, 16, f'{rf_size}x{rf_size}', 
                ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('receptive_field_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_receptive_field() 