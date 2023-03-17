import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def copy_rename(input_dir, output_dir):
    os.mkdir(output_dir)
    root = input_dir

    counter = 0
    for folder in os.listdir(root):
        for subfolder in os.listdir(root + folder):
            folder_name = str(counter).rjust(5, "0")
            dst = "/kaggle/working/dataset/" + folder_name
            src = root + folder + '/' + subfolder + '/'
            shutil.copytree(src, dst)
            counter += 1


def clean_dir(output_dir):
    for file in os.listdir(output_dir):
        shutil.rmtree(output_dir + file)

