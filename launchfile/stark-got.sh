tracker=(stark50 stark101 stark101-got stark50-got stark50 stark101 stark101-got stark50-got)
yaml=(baseline baseline_R101 baseline_R101_got10k_only baseline_got10k_only baseline baseline_R101 baseline_R101_got10k_only baseline_got10k_only)
dataset_name=(got10k_val got10k_val got10k_val got10k_val got10k_val got10k_val got10k_val got10k_val)
dataset=(GOT-10K GOT-10K GOT-10K GOT-10K GOT-10K-C GOT-10K-C GOT-10K-C GOT-10K-C)
dataset_sub=(val val val val val val val val)
for i in ${!tracker[*]};
do
CUDA_VISIBLE_DEVICES=0 python /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/tracking/test.py stark_st ${yaml[$i]} --dataset_name ${dataset_name[$i]} --threads 4 --num_gpus 1 --result_name /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/${dataset[$i]}/results/${tracker[$i]} --datasetpath /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/${dataset[$i]}/${dataset_sub[$i]} 
echo ${tracker[$i]} ${datset[$i]} finished!!!!!!!!!!!!!!!!!!!!!!!
done
