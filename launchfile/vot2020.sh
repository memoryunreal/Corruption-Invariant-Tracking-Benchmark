for i in DiMP18 PrDiMP18 PrDiMP50 ToMP50 ToMP101 eco keeptrack kys ATOM lwl
do
CUDA_VISIBLE_DEVICES=5 /opt/conda/envs/pytracking/bin/vot evaluate --workspace /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/vot2020 $i
echo $i vot2020 finished!!!!!!!!!!!!!!!!!!!!!!!
done
#CUDA_VISIBLE_DEVICES=4 /opt/conda/envs/pytracking/bin/vot evaluate --workspace /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/vot2020 DiMP50
