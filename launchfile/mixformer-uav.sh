tracker=(mixformer22k mixformer22k mixformerL mixformerL mixformer1k mixformer1k mixformer22k-got mixformer22k-got mixformer1k-got mixformer1k-got)
yaml=(baseline baseline baseline_large baseline_large baseline_1k baseline_1k baseline baseline baseline_1k baseline_1k)
dataset_name=(got10k_test got10k_test got10k_test got10k_test got10k_test got10k_test got10k_test got10k_test got10k_test got10k_test)
checkpoint=(mixformer_online_22k.pth.tar mixformer_online_22k.pth.tar mixformerL_online_22k.pth.tar mixformerL_online_22k.pth.tar mixformer_online_1k.pth.tar mixformer_online_1k.pth.tar mixformer_online_22k_got.pth.tar mixformer_online_22k_got.pth.tar mixformer_online_got_1k.pth.tar mixformer_online_got_1k.pth.tar)
dataset=(GOT-10K-C GOT-10K GOT-10K-C GOT-10K GOT-10K-C GOT-10K GOT-10K-C GOT-10K GOT-10K-C GOT-10K)
dataset_sub=(val val val val val val val val val val)
for i in ${!tracker[*]};
do
CUDA_VISIBLE_DEVICES=1,2 python /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/MixFormer/tracking/test.py mixformer_online ${yaml[$i]} --dataset_name ${dataset_name[$i]} --threads 8 --num_gpus 2 --params__search_area_scale 4.55 --params__model ${checkpoint[$i]} --params__max_score_decay 0.98 --result_name /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/${dataset[$i]}/results/${tracker[$i]} --datasetpath /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/${dataset[$i]}/${dataset_sub[$i]} 
echo ${tracker[$i]} ${datset[$i]} finished!!!!!!!!!!!!!!!!!!!!!!!
done
