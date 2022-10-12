#python /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/pytracking/pytracking/run_tracker.py dimp dimp50 --dataset_name got10k_val --gpuid 6 --threads 4 --result_name /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/GOT-10K-C/results/DiMP50 --corp_dataset 0
array=(dimp dimp dimp dimp tomp tomp keep_track keep_track eco kys)
array2=(dimp50 dimp18 prdimp18 prdimp50 tomp50 tomp101 default default_fast default default)
for i in ${!array[*]};
do
python /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/pytracking/pytracking/run_tracker.py ${array[$i]} ${array2[$i]} --dataset_name uav --gpuid 1 --threads 4 --result_name /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/UAV123/results/${array[$i]}_${array2[$i]} --corp_dataset 0 --datasetpath /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/UAV123/
echo $i uav123 finished!!!!!!!!!!!!!!!!!!!!!!!
done
