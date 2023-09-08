import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib import patches

# import imgs_dict.npy for plotting:
image_dict = np.load('imgs_dict.npy', allow_pickle=True).item()
image_dict_ = np.load('imgs_dict.npy', allow_pickle=True).item()

# if 'Cycle' in the dictionary key, change it to cycleGAN:
for key in list(image_dict.keys()):
    if 'Cycle' in key:
        image_dict[key.replace('Cycle', 'cycleGAN')] = image_dict.pop(key)

del image_dict['notes_MB']
del image_dict['wavenumbers']
del image_dict['slice_wavenumbers']

# Ensure all values in the image_dict are numpy arrays
for key, value in image_dict.items():
    if not isinstance(value, np.ndarray):
        raise ValueError(f"The value for key '{key}' is not a numpy array.")

# Custom order for the "Image" column
# custom_order = ['Low SNR', 'Wavelet', 'Cycle', 'High SNR']
custom_order = ['Low SNR', 'Wavelet', 'cycleGAN', 'High SNR']

# Create the grid of images with desired column ordering
fig, axes = plt.subplots(nrows=len(image_dict) // len(custom_order), ncols=len(custom_order), figsize=(10, 8))

# Function to display the image in each grid cell
def plot_image(image, img_label, ax):
    ax.imshow(image.T, cmap='viridis', vmin=0, vmax=1)
    # ax.set_title(img_label, fontsize=8)
    ax.axis('off')

    # Add bold white text in the top left-hand corner
    text_x = 0.05
    text_y = 0.95
    ax.text(text_x, text_y, img_label.split()[-1] + r" cm$^{-1}$",
            transform=ax.transAxes, ha='left', va='top',
            fontsize=12, fontweight='bold', color='white')

    # Highlight specific pixels with an arrow pointing diagonally down
    # if img_label in ['High SNR 1005','High SNR 1336','High SNR 1450','Cycle 1005','Cycle 1336','Cycle 1450','Wavelet 1005','Wavelet 1336','Wavelet 1450']:
    if img_label in ['High SNR 1005','High SNR 1336','High SNR 1450','cycleGAN 1005','cycleGAN 1336','cycleGAN 1450','Wavelet 1005','Wavelet 1336','Wavelet 1450']:

        length = 10
        dx =  - length * np.cos(45 * np.pi / 180)
        dy =  - length * np.sin(45 * np.pi / 180)

        arrow1 = patches.Arrow(45 - dx, 60 - dy, dx, dy, width=5, edgecolor='w', facecolor='w', lw=2)
        ax.add_patch(arrow1)

        arrow2 = patches.Arrow(25 - dx, 17 - dy, dx, dy, width=5, edgecolor='r', facecolor='r', lw=2)
        ax.add_patch(arrow2)

        # # Add text next to the arrow
        # ax.text(arrow_x+12, arrow_y+3, 'A', fontsize=18, color='red', fontweight='bold')

# Plot the images on the grid
for idx, img_type in enumerate(custom_order):
    img_rows = [key for key in image_dict.keys() if key.startswith(img_type)]
    for j, img_key in enumerate(img_rows):
        plot_image(image_dict[img_key], img_key, axes[j, idx])

# Add column headings
for idx, col_heading in enumerate(custom_order):
    axes[0, idx].set_title(col_heading, fontsize=14, fontweight='bold')

# Hide empty subplots
for i in range(len(image_dict) // len(custom_order)):
    for j in range(len(custom_order)):
        if i * len(custom_order) + j >= len(image_dict):
            axes[i, j].axis('off')

plt.subplots_adjust(wspace=0.05, hspace=0.02)  # Adjust spacing between subplots
fig.text(0.0, 1.01, 'A', fontsize=18, color='k', fontweight='bold')

plt.tight_layout()
plt.savefig('HSI_viridis_.pdf', bbox_inches='tight')