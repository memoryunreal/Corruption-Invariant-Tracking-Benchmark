### Training MixFormer-22k
# Stage1: train mixformer without SPM
python tracking/train.py --script mixformer --config baseline --save_dir /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/checkpoints/stage1 --mode multiple --nproc_per_node 8
## Stage2: train mixformer_online, i.e., SPM (score prediction module)
python tracking/train.py --script mixformer_online --config baseline --save_dir /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/checkpoints/stage2 --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL
