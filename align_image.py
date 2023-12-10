import numpy as np
import os

import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import features
from rasterio.mask import mask
from rasterio.windows import Window

def align_images(area_images, destination_crs):
    for img in area_images:
        with rasterio.open(img, "r+") as src:
            transform, width, height = calculate_default_transform(
            src.crs, destination_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': destination_crs,
                           'transform': transform,
                           'width': width,
                           'height': height})
    
            print("Source:", img)
            print(src.crs)
            print(src.bounds)
            print(src.transform)
        
            with rasterio.open(img[:-4]+'_aligned.tif', 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                             source=rasterio.band(src, i),
                             destination=rasterio.band(dst, i),
                             src_transform=src.transform,
                             src_crs=src.crs,
                             dst_transform=transform,
                             dst_crs=destination_crs,
                             resampling=Resampling.nearest)
            
                print("Destination:")
                print(dst.crs)
                print(dst.bounds)
                print(dst.transform)
                print("")
    
# Adapted to our needs from code originally written by Viola Ho.
def cut_images(area_pre_image, area_images, attributes, dataset_dir, file_prefix):   
    with rasterio.open(area_pre_image) as src:
        x_size, y_size = 1024, 1024    
        # Calculate the number of slices in each dimension
        num_slices_x = src.width // x_size
        num_slices_y = src.height // y_size
        
        count = 0
        for j in range(num_slices_y):
            for i in range(num_slices_x):
                count += 1
                
                slice_width = min(x_size, src.width - i * x_size)
                slice_height = min(y_size, src.height - j * y_size)
                
                window = Window(x_size * i, y_size * j, slice_width, slice_height)
                transform = src.window_transform(window)
        
                profile = src.profile
                profile.update({'height': slice_height,
                                'width': slice_width,
                                'transform': transform})

                with rasterio.open(dataset_dir+'pre_img\\'+file_prefix+'_pre_'+(str)(count)+'.tif', 'w', **profile) as dst:
                    pre_data = src.read(window=window)
                    # Pad the slice if its dimensions are smaller than the desired size
                    if slice_width < x_size or slice_height < y_size:
                        pad_width_x = max(0, x_size - slice_width)
                        pad_width_y = max(0, y_size - slice_height)
                        pre_data = np.pad(pre_data, ((0, 0), (0, pad_width_y), (0, pad_width_x)), mode='constant', constant_values=0)
                        
                    dst.write(pre_data)
                    band = src.read(1, window=window)
                    band[np.where(band!=src.nodata)] = 1
            
                    for img, att in zip(area_images, attributes):   
                        with rasterio.open(img, "r+") as shaped:
                            geo_ = []
                            for geometry, raster_value in features.shapes(band, transform=transform):
                                if (raster_value == 1):
                                    geo_.append(geometry)
                
                            out_img, out_transform = mask(dataset=shaped, shapes=geo_, crop=True)
                            
                            if out_img.shape[0] == 1:
                                out_img = np.tile(out_img, (3, 1, 1))
                            print(out_img.shape)
                            
                            with rasterio.open(dataset_dir+'\\'+att+'\\'+file_prefix+'_'+att+'_'+(str)(count)+'.tif', 'w', driver='GTiff', height=out_img.shape[1], width=out_img.shape[2],
                                               count=dst.count, dtype=out_img.dtype, transform=out_transform) as dst:
                                dst.write(out_img)

if __name__ == "__main__":
    cwd = os.getcwd()
    
    area1_pre_image = 'original_imagery\Area1_4-3-2017_Ortho_ColorBalance.tif'
    area2_pre_image = 'original_imagery\Area2_4-3-2017_Ortho_ColorBalance.tif'
    
    area1_post_image = 'original_imagery\Area1_9-2-2017_Ortho_ColorBalance.tif'
    area2_post_image = 'original_imagery\Area2_9-2-2017_Ortho_ColorBalance.tif'
    
    area1_pde_labels = 'PDE_label\PDE_label_Area1_with_background.tif'
    area2_pde_labels = 'PDE_label\PDE_label_Area2_with_background.tif'
    
    area1_elevation = 'meta_attributes\elevation\elevation_Area1.tif'
    area2_elevation = 'meta_attributes\elevation\elevation_Area2.tif'
    
    area1_hand = r'meta_attributes\Height Above Nearest Drainage (HAND)\rem_zeroed_masked_healed_Area1.tif'
    area2_hand = r'meta_attributes\Height Above Nearest Drainage (HAND)\rem_zeroed_masked_healed_Area2.tif'
    
    area1_imperviousness = r'meta_attributes\impervious\nlcd_2016_impervious_l48_Area1.tif'
    area2_imperviousness = r'meta_attributes\impervious\nlcd_2016_impervious_l48_Area2.tif'
    
    area_images = [area1_post_image, area1_pde_labels, area1_elevation, area1_hand, area1_imperviousness, area2_post_image, area2_pde_labels, area2_elevation, area2_hand, area2_imperviousness]
    align_images(area_images, 'EPSG:4326')
    
    area_images = [area1_post_image]

    area1_pre_image = 'original_imagery\Area1_4-3-2017_Ortho_ColorBalance_aligned.tif'
    area2_pre_image = 'original_imagery\Area2_4-3-2017_Ortho_ColorBalance_aligned.tif'
    
    area1_post_image = 'original_imagery\Area1_9-2-2017_Ortho_ColorBalance_aligned.tif'
    area2_post_image = 'original_imagery\Area2_9-2-2017_Ortho_ColorBalance_aligned.tif'
    
    area1_pde_labels = 'PDE_label\PDE_label_Area1_with_background_aligned.tif'
    area2_pde_labels = 'PDE_label\PDE_label_Area2_with_background_aligned.tif'
    
    area1_elevation = 'meta_attributes\elevation\elevation_Area1_aligned.tif'
    area2_elevation = 'meta_attributes\elevation\elevation_Area2_aligned.tif'
    
    area1_hand = r'meta_attributes\Height Above Nearest Drainage (HAND)\rem_zeroed_masked_healed_Area1_aligned.tif'
    area2_hand = r'meta_attributes\Height Above Nearest Drainage (HAND)\rem_zeroed_masked_healed_Area2_aligned.tif'
    
    area1_imperviousness = r'meta_attributes\impervious\nlcd_2016_impervious_l48_Area1_aligned.tif'
    area2_imperviousness = r'meta_attributes\impervious\nlcd_2016_impervious_l48_Area2_aligned.tif'
    
    # Refresh area_images with the new file paths.
    area_images = [area1_post_image, area1_pde_labels, area1_elevation, area1_hand, area1_imperviousness, area2_post_image, area2_pde_labels, area2_elevation, area2_hand, area2_imperviousness]
    area1_images = area_images[0:5]
    area2_images = area_images[5:10]
    
    attributes = ['post_img', 'PDE_labels', 'elevation', 'hand', 'imperviousness']
    
    output_dir = 'dataset\\'
    
    cut_images(area1_pre_image, area1_images, attributes, output_dir, 'Area_1')
    cut_images(area2_pre_image, area2_images, attributes, output_dir, 'Area_2')