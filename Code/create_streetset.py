import numpy as np
from PIL import Image
BASE_DIR = "/data/horse/ws/juul507e-ju_streetsigns/data/"
STREET_DIR = BASE_DIR + "Gstreet/"
TARGET_DIR = BASE_DIR + "streetimages/"

counter = 0
for i in range(1,3001):
    for j in range(1,5):
        filename = str(i).zfill(6) + "_" + str(j)
        im_s = np.array(Image.open(STREET_DIR + filename + ".jpg"))
        im_s_save = Image.fromarray(im_s[:-24:,140:-140])
        im_s_save.save(TARGET_DIR + str(counter).zfill(6) + ".png")
        counter +=1