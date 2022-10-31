import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os

def UAVTESTDataset():
    return UAVTESTDatasetClass().get_sequence_list()


class UAVTESTDatasetClass(BaseDataset):
    """VOT2019 RGBD dataset

    Download the dataset from http://www.votchallenge.net/vot2019/dataset.html"""
    def __init__(self, datasetpath=None):
        super().__init__()
        self.base_path = self.env_settings.uavtest_path
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if datasetpath:
            self.base_path = datasetpath
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name

        anno_path = '{}/{}/{}.txt'.format(self.base_path, sequence_name,'groundtruth')
        #print(anno_path)

        if os.path.exists(str(anno_path)):
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
            except:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

            end_frame = ground_truth_rect.shape[0]
            ground_truth_rect=ground_truth_rect[:,[0,1,2,3]]

        else:
            #print('ptbdataset, no full groundtruth file, use init file')
            anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
            try:
                ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
            except:
                ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)
            #print(ground_truth_rect.shape)
            ground_truth_rect=ground_truth_rect.reshape(1,4)


        frames_path = '{}/{}/color'.format(self.base_path, sequence_path)
        rgb_frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith('jpg') or frame.endswith('png')]
        rgb_frame_list.sort(key=lambda f: int(f[0:8]))
        rgb_frame_list = [os.path.join(frames_path, frame) for frame in rgb_frame_list]
        #print('votddataset, rgb_frame_list[0] %s' % rgb_frame_list[0])

        depth_frames_path='{}/{}/depth'.format(self.base_path, sequence_path)
        depth_frame_list=[frame for frame in os.listdir(depth_frames_path) if frame.endswith('png')]
        depth_frame_list.sort(key=lambda f: int(f[0:8]))
        depth_frame_list = [os.path.join(depth_frames_path, frame) for frame in depth_frame_list]

        if len(ground_truth_rect)==0:
            ground_truth_rect=np.zeros((len(rgb_frame_list),4))

        return Sequence(sequence_name, rgb_frame_list, ground_truth_rect, depth_frame_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        seqlistfile = os.path.join(self.base_path, 'list.txt')
        sequence_list= []
        if os.path.exists(seqlistfile):
            with open(seqlistfile, 'r') as f:
                sequence_list = f.read().splitlines()
        return sequence_list
