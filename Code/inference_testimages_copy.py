import os
import shutil

def copy_images_to_single_folder(source_root, destination_folder):

    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Iterate through all subdirectories in the source root folder
    for root, _, files in os.walk(source_root):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_folder, file)
                
                # Handle potential filename conflicts
                if os.path.exists(destination_file):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(destination_file):
                        destination_file = os.path.join(destination_folder, f"{base}_{counter}{ext}")
                        counter += 1
                
                # Copy the file
                shutil.copy(source_file, destination_file)
                print(f"Copied {source_file} to {destination_file}")

# Absolute paths
source_root = "/data/horse/ws/knoll-traffic_sign_reproduction/atsds_large/test"
# source_root = "data/atsds_large/test"

destination_folder = "/data/horse/ws/knoll-traffic_sign_reproduction/atsds_large/inference/images_to_classify"
# destination_folder = "inference/images_to_classify"

# Execute the function
copy_images_to_single_folder(source_root, destination_folder)
