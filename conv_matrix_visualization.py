import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.colors as mcolors

def visualize_conv_as_matrix():
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # Set up the grid
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1], height_ratios=[1, 1, 1])
    
    # Colors
    input_color = mcolors.to_rgba('lightblue', 0.7)
    filter_color = mcolors.to_rgba('lightgreen', 0.7)
    output_color = mcolors.to_rgba('lightcoral', 0.7)
    skip_color = mcolors.to_rgba('plum', 0.7)
    
    # Define actual dimensions
    H, W = 32, 32  # Input height and width
    C_in = 3       # Input channels (e.g., RGB)
    K = 3          # Kernel size
    C_out = 64     # Output channels
    C_decoder = 32 # Decoder channels
    C_encoder = 64 # Encoder channels
    
    # ===== PART 1: CONVOLUTION AS MATRIX MULTIPLICATION =====
    
    # Input tensor visualization
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.set_title(f'Input Tensor X\nShape: ({H}, {W}, {C_in})', fontsize=12)
    ax_input.set_xlim(0, 5)
    ax_input.set_ylim(0, 5)
    ax_input.grid(True)
    
    # Draw input tensor as a 3D cube
    input_cube = Rectangle((0.5, 0.5), 4, 4, facecolor=input_color, edgecolor='black')
    ax_input.add_patch(input_cube)
    ax_input.text(2.5, 2.5, f'X\n({H}, {W}, {C_in})', ha='center', va='center', fontsize=10)
    
    # Reshaped input matrix
    ax_reshaped = fig.add_subplot(gs[0, 1])
    ax_reshaped.set_title(f'Reshaped Input X\'\nShape: ({H*W}, {C_in})', fontsize=12)
    ax_reshaped.set_xlim(0, 5)
    ax_reshaped.set_ylim(0, 5)
    ax_reshaped.grid(True)
    
    # Draw reshaped matrix
    reshaped_matrix = Rectangle((0.5, 0.5), 4, 4, facecolor=input_color, edgecolor='black')
    ax_reshaped.add_patch(reshaped_matrix)
    ax_reshaped.text(2.5, 2.5, f'X\'\n({H*W}, {C_in})', ha='center', va='center', fontsize=10)
    
    # Filter weights
    ax_filter = fig.add_subplot(gs[0, 2])
    ax_filter.set_title(f'Filter Weights W\nShape: ({K*K*C_in}, {C_out})', fontsize=12)
    ax_filter.set_xlim(0, 5)
    ax_filter.set_ylim(0, 5)
    ax_filter.grid(True)
    
    # Draw filter matrix
    filter_matrix = Rectangle((0.5, 0.5), 4, 4, facecolor=filter_color, edgecolor='black')
    ax_filter.add_patch(filter_matrix)
    ax_filter.text(2.5, 2.5, f'W\n({K*K*C_in}, {C_out})', ha='center', va='center', fontsize=10)
    
    # Output
    ax_output = fig.add_subplot(gs[0, 3])
    ax_output.set_title(f'Output\nShape: ({H*W}, {C_out})', fontsize=12)
    ax_output.set_xlim(0, 5)
    ax_output.set_ylim(0, 5)
    ax_output.grid(True)
    
    # Draw output matrix
    output_matrix = Rectangle((0.5, 0.5), 4, 4, facecolor=output_color, edgecolor='black')
    ax_output.add_patch(output_matrix)
    ax_output.text(2.5, 2.5, f'Output\n({H*W}, {C_out})', ha='center', va='center', fontsize=10)
    
    # Add arrows between matrices
    arrow1 = FancyArrowPatch((4.5, 2.5), (0.7, 2.5), 
                             connectionstyle="arc3,rad=0.2", 
                             arrowstyle='->', color='black')
    ax_reshaped.add_patch(arrow1)
    ax_reshaped.text(2.6, 4.8, 'Reshape', ha='center', fontsize=10)
    
    arrow2 = FancyArrowPatch((4.5, 2.5), (0.7, 2.5), 
                             connectionstyle="arc3,rad=0.2", 
                             arrowstyle='->', color='black')
    ax_filter.add_patch(arrow2)
    ax_filter.text(2.6, 4.8, 'Matrix Multiplication', ha='center', fontsize=10)
    
    # ===== PART 2: CHANNEL INCREASE =====
    
    # Input space
    ax_input_space = fig.add_subplot(gs[1, 0:2])
    ax_input_space.set_title(f'Input Space: Lower-dimensional ({C_in})', fontsize=12)
    ax_input_space.set_xlim(0, 5)
    ax_input_space.set_ylim(0, 5)
    ax_input_space.grid(True)
    
    # Draw input space
    input_space = Rectangle((0.5, 0.5), 4, 4, facecolor=input_color, edgecolor='black')
    ax_input_space.add_patch(input_space)
    ax_input_space.text(2.5, 2.5, f'Input Space\n{C_in} dimensions', ha='center', va='center', fontsize=10)
    
    # Projection
    ax_projection = fig.add_subplot(gs[1, 1:3])
    ax_projection.set_title(f'Projection: W^T · p\nShape: ({K*K*C_in}, {C_out})', fontsize=12)
    ax_projection.set_xlim(0, 5)
    ax_projection.set_ylim(0, 5)
    ax_projection.grid(True)
    
    # Draw projection
    projection = Rectangle((0.5, 0.5), 4, 4, facecolor=filter_color, edgecolor='black')
    ax_projection.add_patch(projection)
    ax_projection.text(2.5, 2.5, f'W^T · p\n({K*K*C_in}, {C_out})', ha='center', va='center', fontsize=10)
    
    # Output space
    ax_output_space = fig.add_subplot(gs[1, 2:4])
    ax_output_space.set_title(f'Output Space: Higher-dimensional ({C_out})', fontsize=12)
    ax_output_space.set_xlim(0, 5)
    ax_output_space.set_ylim(0, 5)
    ax_output_space.grid(True)
    
    # Draw output space
    output_space = Rectangle((0.5, 0.5), 4, 4, facecolor=output_color, edgecolor='black')
    ax_output_space.add_patch(output_space)
    ax_output_space.text(2.5, 2.5, f'Output Space\n{C_out} dimensions', ha='center', va='center', fontsize=10)
    
    # Add arrows
    arrow3 = FancyArrowPatch((4.5, 2.5), (0.7, 2.5), 
                             connectionstyle="arc3,rad=0.2", 
                             arrowstyle='->', color='black')
    ax_projection.add_patch(arrow3)
    ax_projection.text(2.6, 4.8, 'Projection', ha='center', fontsize=10)
    
    arrow4 = FancyArrowPatch((4.5, 2.5), (0.7, 2.5), 
                             connectionstyle="arc3,rad=0.2", 
                             arrowstyle='->', color='black')
    ax_output_space.add_patch(arrow4)
    ax_output_space.text(2.6, 4.8, 'Higher-dimensional space', ha='center', fontsize=10)
    
    # ===== PART 3: SKIP CONNECTIONS =====
    
    # Upsampled features
    ax_upsampled = fig.add_subplot(gs[2, 0])
    ax_upsampled.set_title(f'Upsampled Features U\nShape: ({H*W}, {C_decoder})', fontsize=12)
    ax_upsampled.set_xlim(0, 5)
    ax_upsampled.set_ylim(0, 5)
    ax_upsampled.grid(True)
    
    # Draw upsampled features
    upsampled = Rectangle((0.5, 0.5), 4, 4, facecolor=output_color, edgecolor='black')
    ax_upsampled.add_patch(upsampled)
    ax_upsampled.text(2.5, 2.5, f'U\n({H*W}, {C_decoder})', ha='center', va='center', fontsize=10)
    
    # Skip features
    ax_skip = fig.add_subplot(gs[2, 1])
    ax_skip.set_title(f'Skip Features S\nShape: ({H*W}, {C_encoder})', fontsize=12)
    ax_skip.set_xlim(0, 5)
    ax_skip.set_ylim(0, 5)
    ax_skip.grid(True)
    
    # Draw skip features
    skip = Rectangle((0.5, 0.5), 4, 4, facecolor=skip_color, edgecolor='black')
    ax_skip.add_patch(skip)
    ax_skip.text(2.5, 2.5, f'S\n({H*W}, {C_encoder})', ha='center', va='center', fontsize=10)
    
    # Concatenation
    ax_concat = fig.add_subplot(gs[2, 2])
    ax_concat.set_title(f'Concatenation [U|S]\nShape: ({H*W}, {C_decoder + C_encoder})', fontsize=12)
    ax_concat.set_xlim(0, 5)
    ax_concat.set_ylim(0, 5)
    ax_concat.grid(True)
    
    # Draw concatenation
    concat = Rectangle((0.5, 0.5), 4, 4, facecolor='lightgray', edgecolor='black')
    ax_concat.add_patch(concat)
    
    # Draw the two parts of the concatenation
    concat_part1 = Rectangle((0.5, 0.5), 2, 4, facecolor=output_color, edgecolor='black')
    ax_concat.add_patch(concat_part1)
    
    concat_part2 = Rectangle((2.5, 0.5), 2, 4, facecolor=skip_color, edgecolor='black')
    ax_concat.add_patch(concat_part2)
    
    ax_concat.text(2.5, 2.5, f'[U|S]\n({H*W}, {C_decoder + C_encoder})', ha='center', va='center', fontsize=10)
    
    # Combined convolution
    ax_combined = fig.add_subplot(gs[2, 3])
    ax_combined.set_title(f'Combined Convolution\nW_combined · [U|S]\nShape: ({H*W}, {C_out})', fontsize=12)
    ax_combined.set_xlim(0, 5)
    ax_combined.set_ylim(0, 5)
    ax_combined.grid(True)
    
    # Draw combined convolution
    combined = Rectangle((0.5, 0.5), 4, 4, facecolor=filter_color, edgecolor='black')
    ax_combined.add_patch(combined)
    ax_combined.text(2.5, 2.5, f'W_combined · [U|S]\n({H*W}, {C_out})', ha='center', va='center', fontsize=10)
    
    # Add arrows
    arrow5 = FancyArrowPatch((4.5, 2.5), (0.7, 2.5), 
                             connectionstyle="arc3,rad=0.2", 
                             arrowstyle='->', color='black')
    ax_skip.add_patch(arrow5)
    ax_skip.text(2.6, 4.8, 'Skip Connection', ha='center', fontsize=10)
    
    arrow6 = FancyArrowPatch((4.5, 2.5), (0.7, 2.5), 
                             connectionstyle="arc3,rad=0.2", 
                             arrowstyle='->', color='black')
    ax_concat.add_patch(arrow6)
    ax_concat.text(2.6, 4.8, 'Concatenation', ha='center', fontsize=10)
    
    arrow7 = FancyArrowPatch((4.5, 2.5), (0.7, 2.5), 
                             connectionstyle="arc3,rad=0.2", 
                             arrowstyle='->', color='black')
    ax_combined.add_patch(arrow7)
    ax_combined.text(2.6, 4.8, 'Convolution', ha='center', fontsize=10)
    
    # Add overall title
    fig.suptitle('Convolution as Matrix Multiplication and Skip Connections as Vector Concatenation', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('conv_matrix_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    visualize_conv_as_matrix() 