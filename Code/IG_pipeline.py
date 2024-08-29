import sys
import os
from PIL import Image
import pickle

from ATSDS import ATSDS

import torch
from torchvision import transforms as transforms
## Standard libraries
import os
import json
import math
import random
import numpy as np 
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CUDA_LAUNCH_BLOCKING=1

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

import torchvision.models
import integrated_gradients as int_g


transform_test = transforms.Compose(
    [transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_model(model,optimizer,scheduler,filepath):
    cpt = torch.load(filepath,map_location=torch.device('cpu'))
    model.load_state_dict(cpt['model'])
    optimizer.load_state_dict(cpt['optimizer'])
    scheduler.load_state_dict(cpt['scheduler'])
    return cpt['epoch'], cpt['trainstats'][0], cpt['trainstats'][1]
    
DATASET_PATH = "data"

CHECKPOINT_PATH = "/home/h5/juul507e/Code/model/"
IMAGES_PATH = '/beegfs/ws/1/juul507e-ju_streetsigns/data/atsds_large/train/'

RANDOM_SEED = 1337
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Used for reproducability to fix randomness in some GPU calculations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

from model import get_model

MODEL_NAME = "simple_cnn"

model = get_model(MODEL_NAME, n_classes = 19)
model = model.to(device)

loss_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

INITIAL_LR = 0.1
loss_criterion = loss_criterion.to(device)
optimizer = optim.SGD(model.parameters(),lr=INITIAL_LR, momentum = 0.9,weight_decay=2e-04)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,5000)
running_loss = 0
total = 0
correct = 0
save_osc = 1
epoch = 0
trainloss = []
trainacc = []

epoch,trainloss,trainacc = load_model(model, optimizer, scheduler, "/home/h5/juul507e/Code/model/simple_cnn_1_1.tar")

import cv2


# Returns the Image with the Mask as overlay.
def mask_on_image(mask,img,alpha=0.5):
    heatmap = get_rgb_heatmap(mask)
    img = img.squeeze()
    cam_on_img = (1-alpha)*img + alpha*heatmap
    return np.copy(cam_on_img)

def get_rgb_heatmap(mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    return np.copy(heatmap)

#get a cutout based on a cutoff value
def get_cutoff_area(mask,img,cutoff = 0.5):
    for i in range(3):
        img[:,:,i] = np.where(mask>cutoff,img[:,:,i],0)
    return np.copy(img)

#get a cutout based on a percentage value.
def get_percentage_of_image(image,mask,percentage):
    masked_image = np.zeros_like(image)
    n = mask.size
    sortedvalues = np.sort(mask.flatten('K'))[::-1]
    
    index = int(n/100*percentage)
    index_2 = n//100*percentage
    cutoff = sortedvalues[index]
    for i in range(3):
        masked_image[:,:,i] = np.where(mask-cutoff>0,image[:,:,i],0)
    return masked_image

def normalize_image(img):
    return (img-img.min())/(img.max()-img.min())

def get_input_tensors(image):
    return transform_test(image).unsqueeze(0)



def calculate_ig(directory,runs = 64):
    # Get the list of files in the directory
    
    IMAGES_PATH = directory
    # Define our Categories
    CATEGORIES = sorted(os.listdir(IMAGES_PATH))
    print(CATEGORIES)
    class_to_dataset_class_dict = {}
    for cat in CATEGORIES:
        class_to_dataset_class_dict[cat] = cat
    label_idx_dict = {}
    for count,cat in enumerate(CATEGORIES):
        label_idx_dict[cat] = count

    imagedict = {}
    for cat in CATEGORIES:
        imagedict[cat] = []
        imagelist = os.listdir(IMAGES_PATH + cat + "/")
        for im in imagelist:
            imagedict[cat].append(im)
            
    ig_attribs = {}
    model.to(device)
    for cat in class_to_dataset_class_dict:
        images = imagedict[cat]
        if not os.path.isdir("Code/data/ig/" + MODEL_NAME + "/" + cat):
            os.makedirs("Code/data/ig/" + MODEL_NAME + "/" + cat)
        for imagename in images:
            with open(os.path.abspath(IMAGES_PATH + cat + "/" + imagename), 'rb') as f:
                with Image.open(f) as current_image:
                    current_image_tensor = get_input_tensors(current_image).to(device)
                    ig_attributions = int_g.get_ig_attributions(model,current_image_tensor,label_idx_dict[class_to_dataset_class_dict[cat]],runs)
                    with open("Code/data/ig/" + MODEL_NAME + "/" + cat +  "/" + imagename[:-4] + ".pkl", 'wb') as fp:
                        pickle.dump(ig_attributions, fp)
                        print( imagename + " saved successfully to file")

def main():

    if len(sys.argv) != 2:
        print("Usage: python script.py <image_directory_path>")
        return


    directory_path = sys.argv[1]
    # Open the range of images
    calculate_ig(directory_path)

if __name__ == "__main__":
    main()
