from hashlib import new
import os
import random
import subprocess
import json
# Whole Vide
def bit_error(src,dst,severity):
    c=[100000, 50000, 30000, 20000, 10000][severity-1]
    return_code = subprocess.run(
        ["ffmpeg","-y", "-i", src, "-c", "copy", "-bsf", "noise={}".format(str(c)),
         dst])

    return return_code

def h265_crf(src, dst, severity):
    c = [27, 33, 39, 45, 51][severity - 1]
    return_code = subprocess.call(
        # ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'","-vcodec", "libx265", "-crf", str(c), dst])
        ["ffmpeg", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx264", "-crf", str(c), dst])
    return return_code

def h265_abr(src, dst, severity):
    c = [2, 4, 8, 16, 32][severity - 1]
    result = subprocess.Popen(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)

    data = json.load(result.stdout)

    bit_rate = str(int(float(data['format']['bit_rate']) / c))

    return_code = subprocess.call(
        ["ffmpeg", "-y", "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-b:v", bit_rate, "-maxrate", bit_rate, "-bufsize",
         bit_rate, dst])

    return return_code

# def frame_rate(src,dst,severity):

#     c=[10,8,6,4,2][severity-1]
#     return_code = subprocess.call(
#         ["ffmpeg","-y",  "-i", src,"-vf", "crop='iw-mod(iw,2)':'ih-mod(ih,2)'", "-vcodec", "libx265", "-fps", str(c), dst])
#     return return_code

def img2video(video, imagedir, imageformat):
    jpg = imagedir + '/' + imageformat
    origin_mp4 = video
    video_rate = "25"
    subprocess.call(["ffmpeg", "-f", "image2", "-i", jpg,  "-r", video_rate, origin_mp4, "-y"])
def video2img(video, imagedir, imageformat, seq, corpname, level):
    mp4 = video
    new_data_dir = imagedir
    new_data = new_data_dir+ "/" + "{}_{}_{}_{}".format(seq, corpname, level, imageformat)
    if not os.path.exists(new_data_dir):
        os.makedirs(new_data_dir)
    video_rate = "25"
    subprocess.call(["ffmpeg", "-i", mp4, "-r", video_rate, "-f", "image2", new_data])
corpfuncs = [bit_error, h265_abr, h265_crf]

levels = [1,2,3,4,5]
got10kdataset = "/home/dataset/GOT-10K/val/"
# got10kdataset = "/home/dataset/UAV123/data_seq/UAV123/"
seqlist = []

seqlist = ['GOT-10k_Val_000177', 'GOT-10k_Val_000047']
# imagelist = ['GOT-10k_Val_000177/00000122.jpg', 'GOT-10k_Val_000047/00000047.jpg']
# seqlist= ["wakeboard1"]
for corpf in corpfuncs:
    for seq in seqlist:
        for level in levels:
            # level = 3
            # jpg = "/home/dataset/UAV123/data_seq/UAV123/wakeboard1/%06d.jpg"
            imgformat = "%08d.jpg"
            origin_image_dir = got10kdataset + seq
            origin_mp4 = "/home/dataset/video_corruption/GOT10K/origin/" + "{}.mp4".format(seq)
            new_mp4 = "/home/dataset/video_corruption/GOT10K/corrupt/" + "{}.mp4".format(seq)
            # new_mp4 = "/home/dataset/video_corruption/UAV123/corrupt/"+ "{}.mp4".format(seq)
            new_data_dir = "/home/dataset/GOT-10K-C/tmp/" + corpf.__name__ + "/val/" + seq
            # new_data_dir = "/home/dataset/video_corruption/UAV123/corrupt/" + seq
            # img2video(origin_mp4,origin_image_dir, imgformat)
            # print("{} is ok".format(origin_mp4))
            return_code = corpf(origin_mp4, new_mp4, severity=level)
            # print(return_code)
            video2img(new_mp4, new_data_dir, imgformat, seq, corpf.__name__, level)    
