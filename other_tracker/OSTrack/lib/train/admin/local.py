class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/pretrained_networks'
        self.lasot_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/lasot'
         # prcv competition
        self.competition_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/competition_prcv/data/train'
        self.competition_val_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/competition_prcv/data/val'

        self.got10k_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/got10k/train'
        self.got10k_val_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/got10k/val'
        self.lasot_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/got10k_lmdb'
        self.trackingnet_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/trackingnet_lmdb'
        self.coco_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/coco'
        self.coco_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/vid'
        self.imagenet_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''