tracker=(ostrack256 ostrack384 ostrack384-got ostrack256-got ostrack256 ostrack384 ostrack384-got ostrack256-got)
yaml=(vitb_256_mae_ce_32x4_ep300 vitb_384_mae_ce_32x4_ep300 vitb_384_mae_ce_32x4_got10k_ep100 vitb_256_mae_ce_32x4_got10k_ep100 vitb_256_mae_ce_32x4_ep300 vitb_384_mae_ce_32x4_ep300 vitb_384_mae_ce_32x4_got10k_ep100 vitb_256_mae_ce_32x4_got10k_ep100)
dataset_name=(got10k_val got10k_val got10k_val got10k_val got10k_val got10k_val got10k_val got10k_val)
dataset=(GOT-10K GOT-10K GOT-10K GOT-10K GOT-10K-C GOT-10K-C GOT-10K-C GOT-10K-C)
dataset_sub=(val val val val val val val val)
for i in ${!tracker[*]};
do
CUDA_VISIBLE_DEVICES=1 python /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/OSTrack/tracking/test.py ostrack ${yaml[$i]} --dataset_name ${dataset_name[$i]} --threads 4 --num_gpus 1 --result_name /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/${dataset[$i]}/results/${tracker[$i]} --datasetpath /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/${dataset[$i]}/${dataset_sub[$i]} 
echo ${tracker[$i]} ${datset[$i]} finished!!!!!!!!!!!!!!!!!!!!!!!
done
