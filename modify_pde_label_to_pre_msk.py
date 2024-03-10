import rasterio
import numpy as np
import os

pde_label_paths = 'dataset/PDE_labels'
pre_msk_paths = 'dataset/pre_msk'

pde_label_files = os.listdir(pde_label_paths)
cwd = os.getcwd()

for file in pde_label_files:
    pde_label_path = os.path.join(cwd, pde_label_paths, file)
    pre_msk_path = os.path.join(cwd, pre_msk_paths, file.replace('PDE_labels', 'pre_msk'))
    
    with rasterio.open(pde_label_path) as src:
        data = src.read(1)
        
        modified_data = data.copy()
        modified_data = np.where(modified_data != 4, 255, 4)
        
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)
        
        with rasterio.open(pre_msk_path, 'w', **profile) as dst:
            print("Saving file:", pre_msk_path)
            dst.write(modified_data, 1)