# rough code
import rasterio
from rasterio import windows
import numpy as np
from PIL import Image
import os

def split_tiff(file_path, output_dir, grid_size=(16, 16)):
    with rasterio.open(file_path) as dataset:
        # Calculate the size of each split
        width, height = dataset.width // grid_size[0], dataset.height // grid_size[1]

        # Creating window slices
        window_list = []
        for j in range(grid_size[1]):  # Rows
            for i in range(grid_size[0]):  # Columns
                window = windows.Window(i * width, j * height, width, height)
                window_list.append(window)

        # Process each window
        for idx, window in enumerate(window_list):
            # Read the data in the window
            window_data = dataset.read(window=window)

            # Convert to PIL Image for saving as PNG
            img_data = np.moveaxis(window_data, 0, -1)  # Move channels to last dimension
            img = Image.fromarray(img_data)

            # Save the image
            img.save(os.path.join(output_dir, f'split_image_{idx}.png'), 'PNG')

# Usage
file_path = 'path_to_your_tiff_file.tif'
output_dir = 'path_to_output_directory'
split_tiff(file_path, output_dir)