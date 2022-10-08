import sys
import random
import os
sys.path.append(os.path.abspath('.'))
from nips_corruption import *
from PIL import Image, ImageDraw, ImageFont
import cv2
corruption_function = [gaussian_noise, shot_noise, impulse_noise, defocus_blur,
    glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast,
    elastic_transform, pixelate, jpeg_compression, speckle_noise,
    gaussian_blur, spatter, saturate, rain]
# corruption_function = [spatter]
imagefilepath = '/ssd3/lz/dataset/GOT-10K/GOT-10k_Test_000001/00000001.jpg' 


'''
    visualization
'''
severity_level=3
image_row = 4
image_column = 5
image_save_path = '/home/lz/NIPS2022/corrupt_image.png'
image_size = 256
imagefile = Image.open(imagefilepath)
corrup_imagefile = [Image.fromarray(np.uint8(corrupt_f(imagefile.copy(), severity=severity_level))) for corrupt_f in corruption_function]

to_image = Image.new('RGB', (image_column * image_size, image_row * image_size))
idx = 0
for y in range(1, image_row+1):
    for x in range(1, image_column+1):
        pst_image = corrup_imagefile[idx].resize((image_size, image_size), Image.ANTIALIAS)
        # add text
        draw = ImageDraw.Draw(pst_image)
    
        font = ImageFont.truetype(font='FreeMono.ttf', size=20)
        draw.text(xy=(20,40), text=corruption_function[idx].__name__, fill=(0,0,0), font=font)

        to_image.paste(pst_image, ((x-1)*image_size, (y-1)*image_size))
        idx += 1

to_image.save(image_save_path)