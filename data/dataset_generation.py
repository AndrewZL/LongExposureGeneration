import os
import shutil
import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shutil import copy


def copy_rename(input_dir, output_dir):
    os.mkdir(output_dir)
    root = input_dir

    counter = 0
    for folder in os.listdir(root):
        for subfolder in os.listdir(root + folder):
            folder_name = str(counter).rjust(5, "0")
            dst = os.path.join(output_dir, folder_name)
            src = os.path.join(root, folder, subfolder)
            shutil.copytree(src, dst)
            counter += 1


def clean_dir(output_dir):
    for file in os.listdir(output_dir):
        shutil.rmtree(output_dir + file)


def generate_long_exposure(input_dir, output_dir, thresh=10):
    """
    Generates the newest dataset for training the network based on the original dataset
    by W. Xiong, W. Luo, L. Ma, W. Liu and J. Luo, "Learning to Generate Time-Lapse Videos Using Multi-stage Dynamic Generative Adversarial Networks," in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, USA, 2018 pp. 2364-2373.
    doi: 10.1109/CVPR.2018.00251
    """
    img_folders = []
    for folder in os.listdir(input_dir):
        for subfolder in os.listdir(os.path.join(input_dir, folder)):
            img_folders.append(os.path.join(input_dir, folder, subfolder))

    counter = 0

    for idx, folder in enumerate(img_folders):
        images = sorted(os.listdir(folder))

        # only consider sequences with more than 15 images
        if len(images) > 15:
            q = 0
            gt = []
            prev_frame = np.array(Image.open(os.path.join(folder, images[0]))) / 255.0
            for i in range(1, len(images)):
                if q == 15:
                    break
                next_frame = np.array(Image.open(os.path.join(folder, images[i]))) / 255.0
                # check that the training images are not too similar (contain motion)
                diff = np.absolute(next_frame - prev_frame).mean()
                if diff < 0.03:
                    continue
                else:
                    gt.append(os.path.join(folder, images[i]))
                    q += 1
                    prev_frame = next_frame
                gc.collect()
            # if we have 15 images that are distinct enough, we can add them to the dataset
            if len(gt) == 15:
                # generate ground truth long exposure image: gt_le
                total_seq = np.array([np.array(Image.open(os.path.join(folder, x))) for x in images])
                gt_le = np.average(total_seq, axis=0)
                gt_le = Image.fromarray(np.uint8(gt_le))

                # only save if laplace variance is greater than the threshold
                gray_image = cv2.cvtColor(np.uint8(gt_le), cv2.COLOR_RGB2GRAY)
                laplace_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

                # save the images if the laplace variance is greater than the threshold
                if laplace_var > thresh:
                    gpath = output_dir + 'gt/' + 'gt_{0:06d}/'.format(counter)
                    tpath = output_dir + 'test/' + 'test_{0:06d}/'.format(counter)
                    gspath = output_dir + 'gt/' + 'gt_single_{0:06d}/'.format(counter)
                    os.mkdir(tpath)
                    os.mkdir(gpath)
                    os.mkdir(gspath)

                    # save the first 5 images, which are used as input to the model
                    for j in range(5):
                        tpath + "test_seq_{0:06d}_{0:01d}.jpg".format(counter, j)
                        copy(gt[j], tpath + "test_seq_{:06d}_{:01d}.jpg".format(counter, j))
                        gc.collect()

                    # save the last 10 images, which are used as ground truth
                    for i, path in enumerate(gt[5:]):
                        copy(path, gpath + "gt_seq_{:06d}_{:01d}.jpg".format(counter, i))
                        gc.collect()

                    # save the ground truth long exposure image
                    gt_le.save(gspath + "gt_im_{:06d}.jpg".format(counter))
                    counter += 1
                    gc.collect()

        gc.collect()
