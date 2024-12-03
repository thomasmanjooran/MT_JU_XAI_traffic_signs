import os
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import shutil
import argparse

# Transformation for inference images
transform_inference = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# Function to load the model
def load_model(filepath, model):
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"), weights_only=False)  # Load to CPU or GPU
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint["epoch"], checkpoint["trainstats"]

# Function to predict class for an image
def predict_image(model, image_path, class_names):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform_inference(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# Function to organize images into class folders
def organize_images(model, input_folder, output_folder, class_names):
    # Clear the output folder at the beginning
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)  # Remove all contents in the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Create class folders
    for class_name in class_names:
        class_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(input_folder, filename)
            predicted_class = predict_image(model, file_path, class_names)
            dest_folder = os.path.join(output_folder, predicted_class)
            shutil.copy(file_path, os.path.join(dest_folder, filename))  
            print(f"Copied {filename} to {dest_folder}")



def main(model_full_name):
    # Hardcoded paths
    # input_folder = "inference/images_to_classify"  
    input_folder = "/data/horse/ws/knoll-traffic_sign_reproduction/atsds_large/inference/images_to_classify" 

    # output_folder = "inference/classified_images"  
    output_folder = "/data/horse/ws/knoll-traffic_sign_reproduction/atsds_large/inference/classified_images"  
    


    # Derive model path and model name
    filepath = "model/" + model_full_name + ".tar"
    model_name = "_".join(model_full_name.split("_")[:-2])  # Extract model_name

    # Define class names
    class_names = [f"{i:05d}" for i in range(1, 20)]  # e.g., 00001 to 00019

    # Initialize the model
    model = get_model(model_name=model_name, n_classes=len(class_names)).to(device)

    # Load model weights
    model, epoch, trainstats = load_model(filepath, model)
    model.eval()  # Set model to evaluation mode
    print(f"Loaded model: {model_name} | Epoch: {epoch}")

    # Organize images into class folders
    organize_images(model, input_folder, output_folder, class_names)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_full_name", type=str, required=True, help="Full model name (e.g., simple_cnn_1_1)")
    args = parser.parse_args()

    # Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run the main function
    main(args.model_full_name)
