import os
import numpy as np
from PrRe import PrRe
import logging


class Tracking(object):
    # def __init__(self, name: str, path='/data1/yjy/rgbd_benchmark/all_benchmark/results', reverse=False):
    def __init__(self, name: str, path='/ssd2/lz/rgbd_benchmark/all_benchmark/results', reverse=False):
        self._name = name
        self._path = os.path.join(path, self._name)
        self.reverse = reverse
        self._prre = PrRe()
        # self._seqlist = self.re_list()
        # self._numseq = len(self._seqlist)
        self._lackseq = [] 
        self._votpath = os.path.join(self._path, 'rgbd-unsupervised')

    @property
    def name(self) -> str:
        return self._name
    @property
    def prre(self):
        return self._prre
    @property
    def _votseqlist(self):
        return self.vot_list()

    @property
    def _seqlist(self):
        return self.re_list() 

    @property
    def _numseq(self):
        return len(self._seqlist)

    def lack(self, seq):
        self._lackseq.append(seq)
    def re_list(self):
        seq_list = []
        all_list = os.listdir(self._path)
        for name in all_list:
            if '_001.txt' in name:
                seq = name.split('_001.txt')[0]
                seq_list.append(seq)
            elif '.txt' in name:
                seq = name.split('.txt')[0]
                seq_list.append(seq)
        return seq_list
    def vot_list(self):

        all_list = os.listdir(self._votpath)

        return all_list
    def vot_prebox_conf(self, sequence):
        boxtxt = os.path.join(self._votpath, sequence,'{}_001.txt'.format(sequence))
        try:
            with open(boxtxt, 'r') as f:
                pre_value = np.loadtxt(f, delimiter=',', skiprows=1)
        except:
            logging.debug('use \ t in {}'.format(self._name))
            with open(boxtxt, 'r') as f:
                pre_value = np.loadtxt(f, delimiter='\t', skiprows=1)
        conftxt = os.path.join(self._votpath, sequence, '{}_001_confidence.value'.format(sequence))
        if not os.path.exists(conftxt):
            conftxt = os.path.join(self._votpath, sequence, '{}_001_confidence.txt'.format(sequence))
        
        if not os.path.exists(conftxt):
           confidence = [1 for i in range(len(pre_value))]
        else:
            with open(conftxt, 'r') as f:
                confidence = np.loadtxt(f, skiprows=1)
        return pre_value, confidence 

    def prebox_conf(self, sequence):
            
        boxtxt = os.path.join(self._path, '{}_001.txt'.format(sequence))
        skipframe = 1
        if self._name == 'DSKCF':
            boxtxt = os.path.join(self._path, '{}_001_TargetSize.txt'.format(sequence))
        if self.reverse:
            with open(boxtxt, 'r') as f:
                value0 = f.readlines()
                value0.reverse()
                for value in value0:
                    if float(value.split(",")[0]) == 0.0:
                        skipframe +=1
                    else:
                        break
            pre_value = np.loadtxt(open(boxtxt, 'r').readlines()[:-skipframe], delimiter=',')
            for i in range(skipframe-1):
                pre_value = np.append(pre_value, [[0,0,0,0]], axis=0)
        else:
            try:
                with open(boxtxt, 'r') as f:
                    pre_value = np.loadtxt(f, delimiter=',', skiprows=1)
            except:
                logging.debug('use \ t in {}'.format(self._name))
                with open(boxtxt, 'r') as f:
                    pre_value = np.loadtxt(f, delimiter='\t', skiprows=1)
        conftxt = os.path.join(self._path, '{}_001_confidence.value'.format(sequence))
        if not os.path.exists(conftxt):
            conftxt = os.path.join(self._path, '{}_001_confidence.txt'.format(sequence))
        
        if not os.path.exists(conftxt):
           confidence = [1 for i in range(len(pre_value))]
        else:
            with open(conftxt, 'r') as f:
                confidence = np.loadtxt(f, skiprows=1)
        
        for i in range(len(pre_value) - len(confidence)):
            confidence = np.append(confidence, [0])
  
        return pre_value, confidence
    
    def prebox_speed(self, sequence):
        timetxt = os.path.join(self._path, '{}_001_time.value'.format(sequence))
        if not os.path.exists(timetxt):
            timetxt = os.path.join(self._path, '{}_001_time.txt'.format(sequence))
        if not os.path.exists(timetxt):
            timetxt = os.path.join(self._votpath, '{}/{}_001_time.value'.format(sequence,sequence))
        if not os.path.exists(timetxt):
            timetxt = os.path.join(self._votpath, '{}/{}_time.txt'.format(sequence,sequence))
        flag = False
        
        try:
            with open(timetxt, 'r') as f:
                speed_value = np.loadtxt(f, skiprows=1)
                
        except:
            speed_value = 0
            flag = True
        
        return speed_value, flag       

