import os
import numpy as np
import tifffile
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Stack and combine microscope images from the RSG4.")

# Define command-line arguments
parser.add_argument("--input_directory", type=str, required=True, help="Path to the input directory")
parser.add_argument("--output_directory", type=str, required=True, help="Path to the output directory")
parser.add_argument("--channel", type=str, required=True, help="Channel number to process (e.g., 405)")

# Parse the command-line arguments
args = parser.parse_args()

# Get the input and output directory paths from the command-line arguments
input_directory = args.input_directory
output_directory = args.output_directory

# Set the channel from the command-line arguments
channel = args.channel

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
        x_position = int(tiff_file.split("col")[1].split(".")[0]);
        image_path = os.path.join(images_path, tiff_file)

        if x_position not in x_position_images:
            x_position_images[x_position] = []
        x_position_images[x_position].append(image_path)

# Combine images for each x position with overlap
final_images = []
for x_position, images in x_position_images.items():
    print(images)

    memmaps = [tifffile.memmap(image_path) for image_path in images]

    outname = os.path.join(output_directory,f"ch{channel}_col{x_position:05d}.tif")

    memmap_out = tifffile.memmap(
        outname,
        shape = (len(memmaps),memmaps[0].shape[0],memmaps[0].shape[1]),
        dtype = memmaps[0].dtype
    )

    for z in range(len(memmaps)):
        memmap_out[z,...] = memmaps[z]

    memmap_out.flush()
    del memmap_out

    for mm in memmaps:
        del mm


print(f"Combining complete. The results are saved to {output_directory}")

