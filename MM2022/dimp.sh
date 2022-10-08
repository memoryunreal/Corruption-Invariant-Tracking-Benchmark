for i in gaussian_noise shot_noise impulse_noise glass_blur motion_blur zoom_blur snow frost fog elastic_transform pixelate jpeg_compression speckle_noise gaussian_blur spatter saturate rain
do
echo python /home/MM2022/pytracking/pytracking/run_tracker.py dimp dimp50 --dataset_name got10k_val --gpuid 0 --threads 4 --corp_dataset $i
done