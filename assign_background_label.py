import rasterio
import numpy as np
import os

def set_no_data_value(input_file, output_file, no_data_value, band_value_to_replace):
    with rasterio.open(input_file, "r+") as src:
        src.nodata = no_data_value
        with rasterio.open(output_file, 'w', **src.profile) as dst:
            band = src.read(1)
            band = np.where(band==band_value_to_replace, no_data_value, band)
            dst.write(band, 1)
                
if __name__ == "__main__":
    cwd = os.getcwd()
    input_path = os.path.join(cwd, r"meta_attributes\Height Above Nearest Drainage (HAND)\rem_zeroed_masked_healed_Area1_aligned.tif")
    output_path = os.path.join(cwd, r"meta_attributes\Height Above Nearest Drainage (HAND)\rem_zeroed_masked_healed_Area1_aligned_nodatachanged.tif")
    set_no_data_value(input_path, output_path, -9999, 3.3999999521443642e+38) 