import os
import numpy as np
import tifffile

# Set the input directory path
input_directory = "/scratch.global/tpengo/rib1"  # Replace with the path to your data

# Set the output directory path
output_directory = "/scratch.global/tpengo/rib1_j"  # Replace with the desired output directory

# Set the channel you want to process
channel = "405"  # Replace with the channel number you want to process

# Set the overlap percentage
overlap_percentage = 10

# Get a list of all "layer" directories
layer_directories = [d for d in os.listdir(input_directory) if d.startswith("layer") and os.path.isdir(os.path.join(input_directory,d))]

# Initialize a dictionary to store images for each x position
x_position_images = {}

for layer_dir in layer_directories:
    layer_path = os.path.join(input_directory, layer_dir)
    channel_path = os.path.join(layer_path, channel)
    images_path = os.path.join(channel_path, "images")

    # Get a list of TIFF files starting with "col" and sort them by x position
    tiff_files = [f for f in os.listdir(images_path) if f.startswith("col")]
    tiff_files.sort(key=lambda x: int(x.split("col")[1].split(".")[0]))

    for tiff_file in tiff_files:
        x_position = int(tiff_file.split("col")[1].split(".")[0]); print(x_position)
        image_path = os.path.join(images_path, tiff_file); print(image_path)
        img = tifffile.imread(image_path)

        if x_position not in x_position_images:
            x_position_images[x_position] = []
        x_position_images[x_position].append(img)

# Combine images for each x position with overlap
final_images = []
for x_position, images in x_position_images.items():
    combined_image = np.stack(images)
    #final_images.append(combined_image)
    tifffile.imwrite(os.path.join(output_directory,f"ch{channel}_col{x_position:05d}.tif"),combined_image)

# Combine all x position images into a single image with overlap
#final_image = np.concatenate(final_images,axis=2)

# Save the final image as a TIFF file
#output_file = os.path.join(output_directory, f"combined_channel_{channel}.tiff")
#tifffile.imwrite(output_file,final_image)

#print(f"Combining complete. The result is saved to {output_file}")