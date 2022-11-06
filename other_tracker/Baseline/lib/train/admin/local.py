class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/pretrained_networks'
        self.lasot_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/lasot'
        self.got10k_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/got10k/train'
        self.lasot_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/got10k_lmdb'
        self.trackingnet_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/trackingnet_lmdb'
        self.coco_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/coco'
        self.coco_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/vid'
        self.imagenet_lmdb_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
