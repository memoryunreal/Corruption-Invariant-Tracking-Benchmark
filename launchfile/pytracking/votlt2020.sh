#for i in DiMP18 PrDiMP18 PrDiMP50 ToMP50 ToMP101 eco keeptrack kys ATOM lwl
for i in mixformer mixformerL
do
CUDA_VISIBLE_DEVICES=6 /opt/conda/envs/pytracking/bin/vot evaluate --workspace /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/votlt2020 $i
echo $i votlt2020 finished!!!!!!!!!!!!!!!!!!!!!!!
done
~                                                   
