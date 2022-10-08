import json
import os
import numpy as np

from matplotlib import image
got10k_dir = "/home/dataset/GOT-10K-C/val"
seqlist = os.listdir(got10k_dir)
seqlist.remove("list.txt")
seqlist.sort()

video_dir = seqlist
imagelist = [[] for i in range(len(seqlist))]
init_rect = [[] for i in range(len(seqlist))]
gt_rect = [[] for i in range(len(seqlist))]
seq_dict_value = [[] for i in range(len(seqlist))]
for i in range(len(seqlist)):
    files = os.listdir(os.path.join(got10k_dir, seqlist[i]))
    files.sort()
    # dict imglist
    for fil in files:
        if fil.split(".")[1] == 'jpg':
            imagelist[i].append(os.path.join(seqlist[i], fil))
    # dict init_rect 
    groundtruth = os.path.join(got10k_dir,seqlist[i], "groundtruth.txt")
    with open(groundtruth, 'r') as f:
        value = f.readlines()
        ini = value[0].split("\n")[0]
        # ini = np.array(ini)
        ini = [float(ini.split(",")[0]), float(ini.split(",")[1]), float(ini.split(",")[2]), float(ini.split(",")[3])]
        init_rect[i].append(ini)
        gt_rect[i].append(ini)

        for val in range(len(value)-1):
            gt_rect[i].append([0,0,0,0])

for j in range(len(seqlist)):
    key = ["video_dir", "img_names", "init_rect", "gt_rect"]
    value = [seqlist[j], imagelist[j], init_rect[j], gt_rect[j]]
    seq_dict_value[j] = dict(zip(key, value))

dict_file = dict(zip(seqlist, seq_dict_value))
print("ok")

jsonobj = json.dumps(dict_file)
with open("/home/MM2022/pysot/testing_dataset/json/GOT-10K_val.json", 'w') as f:
    f.write(jsonobj)
    f.close()