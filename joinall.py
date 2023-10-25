import os
import numpy as np
import tifffile

# Create an argument parser
parser = argparse.ArgumentParser(description="Stack and combine microscope images from the RSG4.")

# Define command-line arguments
parser.add_argument("--input_directory", type=str, required=True, help="Path to the input directory")
parser.add_argument("--output_directory", type=str, required=True, help="Path to the output directory")
parser.add_argument("--channel", type=str, required=True, help="Channel number to process (e.g., 405)")
parser.add_argument("--overlap_percentage", type=int, default=10, help="Overlap percentage (default: 10)")

# Parse the command-line arguments
args = parser.parse_args()

# Get the input and output directory paths from the command-line arguments
input_directory = args.input_directory
output_directory = args.output_directory

# Set the channel, and overlap percentage from the command-line arguments
channel = args.channel
overlap_percentage = args.overlap_percentage

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

        if x_position not in x_position_images:
            x_position_images[x_position] = []
        x_position_images[x_position].append(image_path)

# Combine images for each x position with overlap
final_images = []
for x_position, images in x_position_images.items():
    combined_image = np.stack([tifffile.imread(image_path) for image_path in images])
    #final_images.append(combined_image)
    tifffile.imwrite(os.path.join(output_directory,f"ch{channel}_col{x_position:05d}.tif"),combined_image)

# Combine all x position images into a single image with overlap
#final_image = np.concatenate(final_images,axis=2)

# Save the final image as a TIFF file
#output_file = os.path.join(output_directory, f"combined_channel_{channel}.tiff")
#tifffile.imwrite(output_file,final_image)

#print(f"Combining complete. The result is saved to {output_file}")
