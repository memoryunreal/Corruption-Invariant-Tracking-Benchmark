CUDA_VISIBLE_DEVICES=4,5,6,7 python /opt/conda/envs/pytracking/lib/python3.8/site-packages/torch/distributed/launch.py --nproc_per_node 4 --master_port 22921 /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL_v1/lib/train/run_training.py --script stark_st1 --config baseline_got10k_only --save_dir /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Baseline_GL_v1/output/GL_v1 --use_lmdb 0 --script_prv None --config_prv baseline  --distill 0 --script_teacher None --config_teacher None
