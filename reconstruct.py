import numpy as np
import os
from PIL import Image

def reconstruct_image(input_folder, output_path):
    images = os.listdir(input_folder)
    if not images:
        print("No images found in the input folder.")
        return

    # Sort images by name to ensure correct order
    images.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

    # Read first image to get dimensions
    first_image = np.array(Image.open(os.path.join(input_folder, images[0])))
    height, width, channels = first_image.shape
    num_tiles_x = len(set([int(image.split("_")[2].split(".")[0]) for image in images]))
    total_width = num_tiles_x * width
    reconstructed_image = np.zeros((height, total_width, channels), dtype=np.uint8)

    for image_name in images:
        x = int(image_name.split("_")[2].split(".")[0])
        image_path = os.path.join(input_folder, image_name)
        tile = np.array(Image.open(image_path))
        reconstructed_image[:, x:x+width, :] = tile

    # Save the reconstructed image
    print(reconstructed_image.shape)
    reconstructed_image = Image.fromarray(reconstructed_image[:,:6262,:])
    reconstructed_image.save(output_path)

# from PIL import Image
# import os

# def reconstruct_image(input_folder, output_path):
#     images = os.listdir(input_folder)
#     if not images:
#         print("No images found in the input folder.")
#         return

#     # Sort images by name to ensure correct order
#     images.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

#     # Read first image to get dimensions
#     first_image = Image.open(os.path.join(input_folder, images[0]))
#     width, height = first_image.size
#     num_tiles_x = len(set([int(image.split("_")[2].split(".")[0]) for image in images]))
#     total_width = num_tiles_x * (width)
#     reconstructed_image = Image.new('RGB', (total_width, 256))

#     for image_name in images:
#         x = int(image_name.split("_")[2].split(".")[0])
#         image_path = os.path.join(input_folder, image_name)
#         tile = Image.open(image_path)
#         reconstructed_image.paste(tile, (x, 0))

#     # Save the reconstructed image
#     reconstructed_image.save(output_path)

if __name__ == "__main__":
    epochs = 1
    # input_images_folder = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Output_50_Epochs/"
    # output_reconstructed_image_path = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/reconstructed_" + str(epochs) + "_Epoch" + ".png"
    # input_images_folder = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/Label/"
    # output_reconstructed_image_path = "/Users/justindiamond/Documents/Documents/UW-APL/sbp_segmentation/SBP_Dataset_v3/reconstructed_label.png"
    input_images_folder = "E:\Stratigraphy\sbp_segmentation-1/SBP_Dataset_v3/Output_1_Epochs/"
    output_reconstructed_image_path = "E:\Stratigraphy\sbp_segmentation-1/SBP_Dataset_v3/test_label_1_epoch.png"
    reconstruct_image(input_images_folder, output_reconstructed_image_path)