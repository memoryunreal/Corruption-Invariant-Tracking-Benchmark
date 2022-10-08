import random
import os
from turtle import color
import cv2
import os
from mytest.corrupt_transform import Corrupt_Transform as CIT
from PIL import Image
import shutil
from multiprocessing import Pool
import multiprocessing
from time import sleep
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# newdata = "/home/dataset/vot2020-C/sequences/"
# newdata = "/home/dataset/votlt2020-C/sequences/"
# newdata = "/home/dataset/depthtrack-C/sequences/"
# newdata = "/home/dataset/UAV123-C/data_seq/UAV123/"
origindata_dir = "/home/dataset/GOT-10K/val/"

all_corruption = '/home/dataset/NIPS2022_workspace/visualization/'

imagelist = ['GOT-10k_Val_000177/00000122.jpg', 'GOT-10k_Val_000047/00000047.jpg']

corp = CIT()
corp_functions = corp.corruption_function

levels = [1,2,3,4,5,6]

def visualize_corp(image, corp_fun, level, save_dir):
    x = Image.open(os.path.join(origindata_dir, image))
    corp_img = corp_fun(x, severity=level)
    savepath = os.path.join(save_dir, "{}_{}_{}.jpg".format(image.split("/")[0], corp_fun.__name__, level))
    corp_img = Image.fromarray(np.uint8(corp_img))
    corp_img.save(savepath)
    print(savepath)



res = []
for level in levels:
    pool = Pool(processes=multiprocessing.cpu_count())    # set the processes max number 3
    # pool = Pool(processes=1)    # set the processes max number 3
    for cfunc in corp_functions:

        params = (imagelist[1], cfunc, level, all_corruption)
        result = pool.apply_async(visualize_corp, (imagelist[1], cfunc, level, all_corruption))
    pool.close()
    pool.join()

