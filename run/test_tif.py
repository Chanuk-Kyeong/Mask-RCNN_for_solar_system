import cv2

from PIL import Image
im = Image.open('dataset/solar_generation_12cm_final_train/images/S2021SBB05633002.tif')
# im2 = Image.open('dataset/solar_potential_25cm_8234_checked_test/images/S2021AAE002001219.tif')
# im4 = Image.open('dataset/S2021AAA36604004006.tif')
# im5 = Image.open('dataset/S2021AAA36604004012.tif')

# im6 = Image.open('dataset/S2021AAA36604004001.tif')
import numpy as np
 
imarray = np.array(im)
# imarray5 = np.array(im5)
# imarray6 = np.array(im6)

print(imarray.shape)
# print("----------------------------------")

# print(imarray5.shape)
# print("----------------------------------")
# print(imarray6.shape)