import numpy as np
import cv2
from PIL import Image


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
def get_percentage_of_image(image,mask,percentage, fill_value = 0.0):
    masked_image = np.zeros_like(image)
    n = mask.size
    sortedvalues = np.sort(mask.flatten('K'))[::-1]
    
    index = int(n/100*percentage)
    index_2 = n//100*percentage
    cutoff = sortedvalues[index]
    for i in range(3):
        masked_image[:,:,i] = np.where(mask-cutoff>0.0,image[:,:,i],fill_value)
    return masked_image


def get_percentage_of_image_1d(image,mask,percentage, fill_value = 0.0):
    image = normalize_image(image)
    mask = normalize_image(mask)
    masked_image = np.zeros_like(image)
    n = mask.size
    sortedvalues = np.sort(mask.flatten('K'))[::-1]
    
    index = int(n/100*percentage)
    index_2 = n//100*percentage
    cutoff = sortedvalues[index]
    for i in range(3):
        masked_image = np.where(mask-cutoff>0.0,image,fill_value)
    return masked_image

def normalize_image(img):
    return np.nan_to_num((img-img.min())/(img.max()-img.min()), nan=0.0, posinf=0.0,neginf=0.0)

def get_input_tensors(image):
    return transform_test(image).unsqueeze(0)


def get_contained_part(mask1,mask2):
    mask1,mask2 = normalize_image(mask1),normalize_image(mask2)
    return np.array((mask1 == 1.0) & (mask2 == 1.0))
    