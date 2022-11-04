'python -m torch.distributed.launch --nproc_per_node 4 --master_port 40374 lib/train/run_training.py --script ostrack --config vitb_256_mae_ce_32x4_ep300_prcv --save_dir ./output --use_lmdb 0 --script_prv None --config_prv baseline --use_wandb 1 --distill 0 --script_teacher None --config_teacher None'
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 python -m torch.distributed.launch --nproc_per_node 8 --master_port 40374 lib/train/run_training.py --script ostrack --config vitb_256_mae_ce_32x4_ep300_prcv --save_dir ./output --use_lmdb 0 --script_prv None --config_prv baseline --use_wandb 1 --distill 0 --script_teacher None --config_teacher None

# train origin
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 40374 lib/train/run_training.py --script ostrack --config vitb_256_mae_ce_32x4_got10k_ep100_mix --save_dir ./output --use_lmdb 0 --script_prv None --config_prv baseline --use_wandb 1 --distill 0 --script_teacher None --config_teacher None


CUDA_VISIBLE_DEVICES=0,5,6,7 python -m torch.distributed.launch --nproc_per_node 4 --master_port 57890 lib/train/run_training.py --track_mix True --script ostrack --config vitb_256_mae_ce_32x4_got10k_ep100_trackmix --save_dir ./output/mix --use_lmdb 0 --script_prv None --config_prv baseline --use_wandb 1 --distill 0 --script_teacher None --config_teacher None 
