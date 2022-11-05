import os



# newdata = "/home/dataset/GOT-10K-C/firstframe-corrupt/"
# newdata = "/home/dataset4/cvpr2023/GOT-10K-C/val/"
# origindata_dir = "/home/dataset/GOT-10K/val/"

# newdata = "/home/dataset4/cvpr2023/UAV123-C/data_seq/UAV123/"
# origindata_dir = "/home/dataset/UAV123/data_seq/UAV123/"


def vot_dataset():
    newdata = "/home/dataset4/cvpr2023/vot2020-C/sequences/"
    origindata_dir = "/home/dataset/vot2020/sequences/"

    seqlist = os.listdir(origindata_dir)
    seqlist.remove('list.txt') 

    seqlist.sort()


    originseq_color = [os.path.join(origindata_dir,seq,'color') for seq in seqlist] 

    newseq_color = [os.path.join(newdata, seq, 'color') for seq in seqlist]

    originseq_color.sort()
    newseq_color.sort()
    return originseq_color, newseq_color, seqlist

def initialize():
    newdata = "/home/dataset4/cvpr2023/UAV123-C/data_seq/UAV123/"
    origindata_dir = "/home/dataset//UAV123/"

    seqlist = os.listdir(origindata_dir)
    # seqlist.remove('list.txt') # GOT-10K

    seqlist.sort()


    originseq_color = [os.path.join(origindata_dir,seq) for seq in seqlist] 

    newseq_color = [os.path.join(newdata, seq) for seq in seqlist]

    originseq_color.sort()
    newseq_color.sort()
    return originseq_color, newseq_color, seqlist

def got10k():
    newdata = "/home/dataset4/cvpr2023/got10k_trackmix/aug1/train/"
    origindata_dir = "/home/dataset/GOT-10K/train/"

    seqlist = os.listdir(origindata_dir)
    seqlist.remove('list.txt') # GOT-10K

    seqlist.sort()
    seqlist = seqlist[:4000]

    originseq_color = [os.path.join(origindata_dir,seq) for seq in seqlist] 

    newseq_color = [os.path.join(newdata, seq) for seq in seqlist]

    originseq_color.sort()
    newseq_color.sort()
    return originseq_color, newseq_color, seqlist

def checkfiles(i):

    colorfiles = os.listdir(originseq_color[i])

    newcolorfiles = os.listdir(newseq_color[i])

    if not len(colorfiles) == len(newcolorfiles):
        print(newseq_color[i], len(newcolorfiles), originseq_color[i], len(colorfiles))




originseq_color, newseq_color, seqlist = got10k()
# originseq_color, newseq_color, seqlist = vot_dataset()
# originseq_color, newseq_color, seqlist = initialize()



for i in range(len(seqlist)):
    checkfiles(i)

print("all sequences checked")


