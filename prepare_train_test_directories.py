import os
import random
import shutil
import re

# Function to extract area and ending number from filename
def extract_numbers(filename):
    # Regular expression to match the area number and the ending number
    match = re.search(r"Area_(\d+)_.*_(\d+)", filename)
    if match:
        # Return a tuple of area number and ending number as integers
        return (int(match.group(1)), int(match.group(2)))
    else:
        # Return a tuple of zeros if the pattern does not match
        return (0, 0)
    
cwd = os.getcwd()

pre_img_path = 'dataset\pre_img'
post_img_path = 'dataset\post_img'
pde_labels_path = 'dataset\PDE_labels'
pre_msk_path = 'dataset\pre_msk'

elevation_path = 'dataset\elevation'
hand_path = 'dataset\hand'
imperviousness_path = 'dataset\imperviousness'
distance_to_coast_path = 'dataset\distance_to_coast'
distance_to_stream_path = 'dataset\distance_to_stream'

rain_824_path = r'dataset\rain_824'
rain_825_path = r'dataset\rain_825'
rain_826_path = r'dataset\rain_826'
rain_827_path = r'dataset\rain_827'
rain_828_path = r'dataset\rain_828'
rain_829_path = r'dataset\rain_829'
rain_830_path = r'dataset\rain_830'

stream_elev_824_path = 'dataset\stream_elev_824'
stream_elev_825_path = 'dataset\stream_elev_825'
stream_elev_826_path = 'dataset\stream_elev_826'
stream_elev_827_path = 'dataset\stream_elev_827'
stream_elev_828_path = 'dataset\stream_elev_828'
stream_elev_829_path = 'dataset\stream_elev_829'
stream_elev_830_path = 'dataset\stream_elev_830'

pre_img_train_path = r'dataset\training\pre_img'
post_img_train_path = r'dataset\training\post_img'
pde_labels_train_path = r'dataset\training\PDE_labels'
pre_msk_train_path = r'dataset\training\pre_msk'

elevation_train_path = r'dataset\training\elevation'
hand_train_path = r'dataset\training\hand'
imperviousness_train_path = r'dataset\training\imperviousness'
distance_to_coast_train_path = r'dataset\training\distance_to_coast'
distance_to_stream_train_path = r'dataset\training\distance_to_stream'

rain_824_train_path = r'dataset\training\rain\824'
rain_825_train_path = r'dataset\training\rain\825'
rain_826_train_path = r'dataset\training\rain\826'
rain_827_train_path = r'dataset\training\rain\827'
rain_828_train_path = r'dataset\training\rain\828'
rain_829_train_path = r'dataset\training\rain\829'
rain_830_train_path = r'dataset\training\rain\830'

stream_elev_824_train_path = r'dataset\training\stream_elev\824'
stream_elev_825_train_path = r'dataset\training\stream_elev\825'
stream_elev_826_train_path = r'dataset\training\stream_elev\826'
stream_elev_827_train_path = r'dataset\training\stream_elev\827'
stream_elev_828_train_path = r'dataset\training\stream_elev\828'
stream_elev_829_train_path = r'dataset\training\stream_elev\829'
stream_elev_830_train_path = r'dataset\training\stream_elev\830'

pre_img_test_path = r'dataset\testing\pre_img'
post_img_test_path = r'dataset\testing\post_img'
pde_labels_test_path = r'dataset\testing\PDE_labels'
pre_msk_test_path = r'dataset\testing\pre_msk'

elevation_test_path = r'dataset\testing\elevation'
hand_test_path = r'dataset\testing\hand'
imperviousness_test_path = r'dataset\testing\imperviousness'
distance_to_coast_test_path = r'dataset\testing\distance_to_coast'
distance_to_stream_test_path = r'dataset\testing\distance_to_stream'

rain_824_test_path = r'dataset\testing\rain\824'
rain_825_test_path = r'dataset\testing\rain\825'
rain_826_test_path = r'dataset\testing\rain\826'
rain_827_test_path = r'dataset\testing\rain\827'
rain_828_test_path = r'dataset\testing\rain\828'
rain_829_test_path = r'dataset\testing\rain\829'
rain_830_test_path = r'dataset\testing\rain\830'

stream_elev_824_test_path = r'dataset\testing\stream_elev\824'
stream_elev_825_test_path = r'dataset\testing\stream_elev\825'
stream_elev_826_test_path = r'dataset\testing\stream_elev\826'
stream_elev_827_test_path = r'dataset\testing\stream_elev\827'
stream_elev_828_test_path = r'dataset\testing\stream_elev\828'
stream_elev_829_test_path = r'dataset\testing\stream_elev\829'
stream_elev_830_test_path = r'dataset\testing\stream_elev\830'

pre_img_val_path = r'dataset\val\pre_img'
post_img_val_path = r'dataset\val\post_img'
pde_labels_val_path = r'dataset\val\PDE_labels'
pre_msk_val_path = r'dataset\val\pre_msk'

elevation_val_path = r'dataset\val\elevation'
hand_val_path = r'dataset\val\hand'
imperviousness_val_path = r'dataset\val\imperviousness'
distance_to_coast_val_path = r'dataset\val\distance_to_coast'
distance_to_stream_val_path = r'dataset\val\distance_to_stream'

rain_824_val_path = r'dataset\val\rain\824'
rain_825_val_path = r'dataset\val\rain\825'
rain_826_val_path = r'dataset\val\rain\826'
rain_827_val_path = r'dataset\val\rain\827'
rain_828_val_path = r'dataset\val\rain\828'
rain_829_val_path = r'dataset\val\rain\829'
rain_830_val_path = r'dataset\val\rain\830'

stream_elev_824_val_path = r'dataset\val\stream_elev\824'
stream_elev_825_val_path = r'dataset\val\stream_elev\825'
stream_elev_826_val_path = r'dataset\val\stream_elev\826'
stream_elev_827_val_path = r'dataset\val\stream_elev\827'
stream_elev_828_val_path = r'dataset\val\stream_elev\828'
stream_elev_829_val_path = r'dataset\val\stream_elev\829'
stream_elev_830_val_path = r'dataset\val\stream_elev\830'

removals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 27, 28, 29, 33, 34, 35, 43, 44, 48, 49, 50,
            59, 63, 64, 65, 69, 83, 84, 85, 86, 99, 100, 113, 114, 115, 116, 117, 120, 130, 131, 132, 135, 136, 137, 138,
            150, 151, 152, 153, 165, 166, 167, 168, 180, 181, 182, 183, 190, 191, 192, 193, 194, 195, 196, 197, 198, 207,
            208, 209, 210, 211, 212, 213, 224, 225, 226, 227, 228, 240, 241, 242, 243, 252, 253, 254, 255, 256, 257, 258,
            263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287,
            288, 292, 293, 294, 295, 298, 299, 303, 304, 305, 313, 324, 326, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345,
            346, 347, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 365, 366, 367, 368, 369, 370, 371, 372, 373,
            378, 379, 380, 381, 382, 383, 384, 385, 386, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 404, 405,
            406, 407, 408, 409, 410, 411, 412, 417, 418, 419, 420, 421, 422, 423, 424, 425, 430, 431, 432, 433, 434, 435,
            436, 437, 443, 444, 445, 446, 447, 448, 449, 455, 456, 457, 458, 459, 460, 461, 468, 469, 470, 471, 472, 473,
            481, 482, 483, 484, 494, 495, 496, 506, 507, 508, 518, 519, 520, 532, 533, 534, 535, 536, 540, 541, 542, 543, 544, 545,
            546, 547, 548, 549, 553, 554, 555, 556, 557, 558, 559, 569, 570, 571, 572, 584, 585, 591, 592, 595, 596, 597, 598]

#removal_directories = [pre_img_path, post_img_path, pde_labels_path, elevation_path, hand_path, imperviousness_path, distance_to_coast_path, distance_to_stream_path,
#                       rain_824_path, rain_825_path, rain_826_path, rain_827_path, rain_828_path, rain_829_path, rain_830_path,
#                       stream_elev_824_path, stream_elev_825_path, stream_elev_826_path, stream_elev_827_path, stream_elev_828_path, stream_elev_829_path, stream_elev_830_path]

move_directories = [pre_img_path, post_img_path, pde_labels_path, elevation_path, hand_path, imperviousness_path, distance_to_coast_path, distance_to_stream_path,
                    rain_824_path, rain_825_path, rain_826_path, rain_827_path, rain_828_path, rain_829_path, rain_830_path,
                    stream_elev_824_path, stream_elev_825_path, stream_elev_826_path, stream_elev_827_path, stream_elev_828_path, stream_elev_829_path, stream_elev_830_path]

train_directories = [pre_img_train_path, post_img_train_path, pde_labels_train_path, elevation_train_path, hand_train_path, imperviousness_train_path, distance_to_coast_train_path, distance_to_stream_train_path,
                     rain_824_train_path, rain_825_train_path, rain_826_train_path, rain_827_train_path, rain_828_train_path, rain_829_train_path, rain_830_train_path,
                     stream_elev_824_train_path, stream_elev_825_train_path, stream_elev_826_train_path, stream_elev_827_train_path, stream_elev_828_train_path, stream_elev_829_train_path, stream_elev_830_train_path]

test_directories = [pre_img_test_path, post_img_test_path, pde_labels_test_path, elevation_test_path, hand_test_path, imperviousness_test_path, distance_to_coast_test_path, distance_to_stream_test_path,
                    rain_824_test_path, rain_825_test_path, rain_826_test_path, rain_827_test_path, rain_828_test_path, rain_829_test_path, rain_830_test_path,
                    stream_elev_824_test_path, stream_elev_825_test_path, stream_elev_826_test_path, stream_elev_827_test_path, stream_elev_828_test_path, stream_elev_829_test_path, stream_elev_830_test_path]

val_directories = [pre_img_val_path, post_img_val_path, pde_labels_val_path, elevation_val_path, hand_val_path, imperviousness_val_path, distance_to_coast_val_path, distance_to_stream_val_path,
                   rain_824_val_path, rain_825_val_path, rain_826_val_path, rain_827_val_path, rain_828_val_path, rain_829_val_path, rain_830_val_path,
                   stream_elev_824_val_path, stream_elev_825_val_path, stream_elev_826_val_path, stream_elev_827_val_path, stream_elev_828_val_path, stream_elev_829_val_path, stream_elev_830_val_path]

delete_files = True
generate_new_test_indices = True
move_files_to_train_test = True

with open('log.txt', 'w') as log:
    if (delete_files):
        for i in range(len(move_directories)):
            all_files = sorted(os.listdir(os.path.join(cwd, move_directories[i])), key=extract_numbers)
            file_count = len(all_files)

            for j in range(file_count):
                if j in removals and os.path.exists(os.path.join(cwd, move_directories[i], all_files[j])):
                    print(f"Deleting: {all_files[j]}")
                    log.write(f"Deleting: {all_files[j]}\n")
                    os.remove(os.path.join(cwd, move_directories[i], all_files[j]))

    if (generate_new_test_indices):
        file_count = len(os.listdir(os.path.join(cwd, move_directories[0])))
        test_val_random_indices = random.sample(range(file_count), int(file_count * 0.3))
        
        midpoint = len(test_val_random_indices) // 2 #Split the list into two lists, one for validation and one for testing.
        test_random_indices = test_val_random_indices[:midpoint]
        val_random_indices = test_val_random_indices[midpoint:]
        
        with open('test_indices.txt', 'w') as indices:
            for i in test_random_indices:
                indices.write(str(i) + "\n")
        with open('val_indices.txt', 'w') as indices:
            for i in val_random_indices:
                indices.write(str(i) + "\n")
        
    if (move_files_to_train_test):
        test_random_indices = []
        val_random_indices = []
        
        with open('test_indices.txt', 'r') as indices:
            for i in indices:
                test_random_indices.append(int(i.strip()))
        with open('val_indices.txt', 'r') as indices:
            for i in indices:
                val_random_indices.append(int(i.strip()))
    
        for i in range(len(move_directories)):
            all_files = sorted(os.listdir(os.path.join(cwd, move_directories[i])), key=extract_numbers)
            file_count = len(all_files)
        
            for j in range(file_count):
                if j in test_random_indices and os.path.exists(os.path.join(cwd, test_directories[i])):
                    print(f"Moving: {all_files[j]} to {test_directories[i]}")
                    log.write(f"Moving: {all_files[j]} to {test_directories[i]}\n")
                    shutil.move(os.path.join(cwd, move_directories[i], all_files[j]), os.path.join(cwd, test_directories[i]))
                elif j in val_random_indices and os.path.exists(os.path.join(cwd, val_directories[i])):
                    print(f"Moving: {all_files[j]} to {val_directories[i]}")
                    log.write(f"Moving: {all_files[j]} to {val_directories[i]}\n")
                    shutil.move(os.path.join(cwd, move_directories[i], all_files[j]), os.path.join(cwd, val_directories[i]))
                elif os.path.exists(os.path.join(cwd, train_directories[i])):
                    print(f"Moving: {all_files[j]} to {train_directories[i]}")
                    log.write(f"Moving: {all_files[j]} to {train_directories[i]}\n")
                    shutil.move(os.path.join(cwd, move_directories[i], all_files[j]), os.path.join(cwd, train_directories[i]))