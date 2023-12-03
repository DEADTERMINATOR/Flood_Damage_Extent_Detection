# rough code
import rasterio
from rasterio import windows
import numpy as np
from PIL import Image
import os

def normalize(array):
    """Normalize the array to 0-255 and convert to uint8."""
    array = array.astype(np.float32)
    min_val = array.min()
    max_val = array.max()
    if max_val - min_val > 1e-5:  # Ensure the denominator is not too small
        array -= min_val
        array /= max_val - min_val
        array *= 255
    else:
        array = np.zeros(array.shape)  # Set to zero if there's no variation
    return array.astype(np.uint8)

def split_tiff(file_path, output_dir, grid_size=(16, 16)):
    with rasterio.open(file_path) as dataset:
        # Calculate the size of each split
        width, height = dataset.width // grid_size[0], dataset.height // grid_size[1]

        # Creating window slices
        window_list = []
        for j in range(grid_size[1]):
            for i in range(grid_size[0]):
                window = windows.Window(i * width, j * height, width, height)
                window_list.append(window)

        # Process each window
        for idx, window in enumerate(window_list):
            # Read the data in the window
            window_data = dataset.read(window=window)

            # Normalize the data
            if dataset.count == 1:  # Single band
                window_data = normalize(window_data[0])
                img = Image.fromarray(window_data, 'L')
            else:  # Multiband
                normalized_data = [normalize(band) for band in window_data]
                window_data = np.stack(normalized_data, axis=-1)
                img = Image.fromarray(window_data, 'RGB')

            # Save the image
            img.save(os.path.join(output_dir, f'split_image_{idx}.png'), 'PNG')

# Usage
file_path = 'align&split//align//nlcd_2016_impervious_l48_Area2_aligned.tif'
output_dir = 'align&split//split//nlcd_2016_impervious_l48_Area2_split//'
split_tiff(file_path, output_dir)