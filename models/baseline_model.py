import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from PIL import Image
import os
import math
import cv2 as cv




# transform = torchvision.transforms.Compose([torchvision.transforms.Resize(640,360),torchvision.transforms.ToTensor()])
# full_data = torchvision.datasets.ImageFolder(root = 'testingproj', transform = transform)
# print(full_data.size())

direc = "/Users/Shenshen/Desktop/Daniel_Peishuo/University_of_Toronto/Winter2023/APS360/data"


images = [[]]
listy = os.listdir(direc)
offset = 0
for gawk,i in enumerate(listy):
  keep = 0
  # print("here:" + str(gawk))
  if i == ".DS_Store" or i == "baseline-model.log":
    continue
  posy = os.listdir(direc + "/" + i)
  if images[-1]!= []:
    # print("here")
    images.append([])
  # print(posy)
  for j in posy:
    if j == ".DS_Store":
      continue
    money = os.listdir(direc + "/" + i + "/" + j)
    if images[-1]!= []:
      # print("here")
      images.append([])
    for k in money:
      if k == ".DS_Store":
        continue
      images[-1].append(Image.open(direc + "/" + i + "/" + j + "/" + k))

# print(images)



def compare_images(img1, img2, check = False):

  # Convert images to numpy arrays
  if check:
    arr1 = np.array(img1)*255
    arr2 = np.array(img2)*255
  else:
    arr1 = np.array(img1)
    arr2 = np.array(img2)

  # Compare the two arrays

  diffy = arr2 - arr1

  # Print the pixels that have changed
  return diffy



def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    blue = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)

def shift_image(image, shift):
    
    return np.roll(np.roll(image, shift[0], axis=0), shift[1], axis=1)
count = 0

for j in images:
  count += 1
  pixel_diffs = []
  pos_change = []

  for pos,i in enumerate(j[:-1]):
    pixel_diffs.append(compare_images(i, j[pos+1], True))
  for posy,i in enumerate(pixel_diffs[:-1]):
    derivative = rgb2gray(compare_images(i, pixel_diffs[posy+1])) #numpy array of differences in changes
    size = np.argsort((derivative.reshape(-1)))[:10]
    average, sum = 0,0

    for num in size:
      average += derivative[num//640][num%360]
      sum += derivative[num//640][num%360]*num

    if math.isnan(sum/average):
      sum = 0
    else:
      sum = int(sum/average)
    position = (sum//640, sum%360)
    pos_change.append(position)
  delta = (0,0)
  for posy,i in enumerate(pos_change[:-1]):
    delta = (delta[0]+pos_change[posy+1][0] - pos_change[posy][0], delta[0] + pos_change[posy+1][1] - pos_change[posy][1])

  mask = np.clip(pixel_diffs[-1]*10000, 0, 255)
  blurred_img = cv.blur(pixel_diffs[-1], (5,5))





  kernel_size = 30
  
  # Create the vertical kernel.
  kernel_v = np.zeros((kernel_size, kernel_size))
    
  # Create a copy of the same for creating the horizontal kernel.
  kernel_h = np.copy(kernel_v)
    
  # Fill the middle row with ones.
  kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
  kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    
  # Normalize.
  kernel_v /= kernel_size
  kernel_h /= kernel_size
  
  image = np.array(j[-1])
  Image.fromarray(image.astype(np.uint8), mode = "RGB").show()
  if abs(delta[0]) > abs(delta[1]):
    posi = 0
  else:
    posi = 1
  # Apply the vertical kernel.
  if posi == 1:
    for i in range (delta[0]//delta[1]*10):
      image = cv.filter2D(image, -1, kernel_v)
    for i in range (20):
      mask = cv.filter2D(mask, -1, kernel_v)
    for i in range (5):
      image = cv.filter2D(image, -1, kernel_h)
      mask = cv.filter2D(mask, -1, kernel_h)
      
    
  # Apply the horizontal kernel.
  if posi == 0:
    for i in range (delta[1]//delta[0]*10):
      image = cv.filter2D(image, -1, kernel_h)
    for i in range (20):
      mask = cv.filter2D(mask, -1, kernel_h)
    for i in range (5):
      image = cv.filter2D(image, -1, kernel_v)
      mask = cv.filter2D(mask, -1, kernel_v)

  image = cv.blur(image, (5,5))
  
  print(mask)
  mask = np.clip(mask, 0, 200)
  
  # out = np.where(mask==np.array([255, 0, 0]) , image, np.array(j[-1]))
  # out = np.where(mask==np.array([0, 255, 0]) , image, out)
  # out = np.where(mask==np.array([0, 0, 255]) , image, out)
  out = np.where(mask==np.array([200, 200, 200]) , image, np.array(j[-1]))
  out = Image.fromarray(out.astype(np.uint8), mode = "RGB")
  out.show()
  if count > 3:
    break
  # change = pixel_diffs[-1]
  # next_img = pixel_diffs[-1]
  # for i in range (1000):
  #   change = change + shift_image(next_img, delta)
  #   next_img = shift_image(next_img, delta)
  # print(change- pixel_diffs[-1])

  # # change = cv.blur(change, (5,5))

  # final = np.array(j[-1])*255 + (change/1000).astype(np.uint8)

  # fin = Image.fromarray(final, mode = "RGB")
  # print(fin)
  # fin.show()
  # before = j[-1]
  # before.show()
  # if count == 1:
  #   break


  

    # print(sum)
  

    

# def find_dir_change(pixel_diffs):
#   for i in pixel_diffs:
#     for 

# x = np.array([[[1,2,3,4,5],[6,7,8,9,11]],[[1,2,3,4,5],[6,7,8,9,10]]])
# print(shift_image(x, (1,2)))



