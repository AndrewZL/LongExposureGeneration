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


def generate_long_exposure(input_dir, output_dir):
    thresh = 10

    os.mkdir(output_dir)
    root = input_dir

    for folder in os.listdir(root):
        for subfolder in os.listdir(root + folder):
            savename = subfolder + '.jpg'
            if savename in output_dir:
                continue
            filenames = os.listdir(root + folder + '/' + subfolder)
            prefix = root + folder + '/' + subfolder + '/'
            photo_seq = np.array([np.array(Image.open(prefix + fname)) for fname in filenames])

            mean_frame = np.array(np.mean(photo_seq, axis=0), dtype=np.uint8)
            long_exposure = Image.fromarray(mean_frame)

            # only save if laplace variance is greater than the threshold
            gray_image = cv2.cvtColor(mean_frame, cv2.COLOR_RGB2GRAY)
            laplace_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

            if laplace_var > thresh:
                long_exposure.save(f"{output_dir}/{subfolder}.jpg")

