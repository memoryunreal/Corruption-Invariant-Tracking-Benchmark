import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from os.path import exists
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import math
import multiprocessing
seq_path = '/ssd3/lz/MM2022/dataset/depthtrack/'
from multiprocessing import  Process
import logging
log_path = os.path.join(os.path.split(os.path.realpath(__file__))[0], './prompt.log')
logging.basicConfig(filename=log_path, level=logging.DEBUG, filemode='w', format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%Y-%d-%m %H:%M:%S')

class PromptGeneration(Process):
    def __init__(self,seqpathlist):
        super(PromptGeneration,self).__init__()
        self.seqpathlist = seqpathlist

    def run(self):
        for id, seqpath in enumerate(self.seqpathlist):
            color_path = os.path.join(seqpath, 'color')
            newcolor_path = os.path.join(seqpath, 'newcolor_blend_0.05_depth_5000') 
            depth_path = os.path.join(seqpath, 'depth')
            colormap_path = os.path.join(seqpath, 'colormap')
            # if os.path.exists(newcolor_path) or os.path.exists(colormap_path):
            #     if len(os.listdir(colormap_path)) == len(os.listdir(depth_path)):
            #         logging.info("{} complete".format(seqpath))
            #         continue

            # sequence file rewrite
            sequencefile = os.path.join(seqpath, 'sequence')
            with open(sequencefile, 'r') as f:
                value = f.readlines()
            with open(sequencefile, 'w') as f:
                suffixname = value[0].split(".")[-1]
                if suffixname=='png\n':
                    linevalue = 'channels.color=color/%08d.jpg\n'
                    value[0] = linevalue
                for index, line in enumerate(value):
                    f.writelines(value[index])

            print("{} start".format(seqpath))
            #color_path = '/home/yjy/vot2021/sequences/%s/color/'%seq
            # color_list = os.listdir(color_path)
            # color_list.sort()
            # depth_list = os.listdir(depth_path)
            # depth_list.sort()
            # num_imgs = len(color_list)
            # depth_threshold = 5000
                    
            # if not os.path.exists(colormap_path) or not os.path.exists(newcolor_path):
            #     os.makedirs(colormap_path)
            #     os.makedirs(newcolor_path)
            # else:
            #     # continue
            #     logging.info('{} or {} exists'.format(colormap_path, newcolor_path))
            # for i in range(num_imgs):

            #     # convert depth to colormap
            #     dp =  os.path.join(depth_path, depth_list[i])
            #     dp = cv2.imread(dp, -1)
            #     try:
            #         dp[dp>depth_threshold] = depth_threshold # ignore some large values,
            #     except:
            #         dp = dp
            #     dp = cv2.normalize(dp, None, 0, 255, cv2.NORM_MINMAX)
            #     dp = cv2.applyColorMap(np.uint8(dp), cv2.COLORMAP_JET)

            #     filename = '%08d.png'%(i+1)
            #     cv2.imwrite(os.path.join(colormap_path, filename),dp)

            #     # blend colormap and color
            #     color_file = os.path.join(color_path, color_list[i])
            #     colorf = Image.open(color_file)
            #     colorf = colorf.convert('RGBA')
            #     colormapfile = os.path.join(colormap_path, filename)
            #     colormapf = Image.open(colormapfile)
            #     colormapf = colormapf.convert('RGBA')
            #     blend_img = Image.blend(colorf, colormapf, 0.05)
            #     #img.show()
                
            #     save = os.path.join(newcolor_path, filename)
            #     blend_img.save(save)
            
            # dir rename: color -> oldcolor newcolor -> color
            
            olddir = os.path.join(self.seqpathlist[id], 'oldcolor')
            # os.rename(color_path, olddir)
            os.rename(color_path, newcolor_path)
            os.rename(olddir, color_path)
            # os.rename(newcolor_path, color_path)
            
            # logging.info('{}'.format(seqpath, 'complete'))


if __name__ == '__main__':
    dirpath = '/ssd3/lz/MM2022/dataset/cdtb/'
    seqlist = os.listdir(dirpath)
    seqlist.remove('list.txt')
    # seqlist = ['flag_indoor']
    # process_num = int(multiprocessing.cpu_count() / 2)
    process_num = 10
    seq_list = [os.path.join(dirpath, i) for i in seqlist]
    splitlist = np.array_split(np.array(seq_list), process_num)
    process_list = []

    for i in range(process_num):

        p = PromptGeneration(splitlist[i].tolist()) #实例化进程对象
        p.start()
        process_list.append(p)

    for i in range(len(process_list)):
        process_list[i].join()
        # process_

    print('结束测试')


        