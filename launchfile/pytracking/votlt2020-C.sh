#for i in DiMP18 PrDiMP18 PrDiMP50 ToMP50 ToMP101 eco keeptrack kys ATOM lwl
for i in mixformer mixformerL mixformer1k
do
CUDA_VISIBLE_DEVICES=7 /opt/conda/envs/pytracking/bin/vot evaluate --workspace /home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/votlt2020-C $i
echo $i votlt2020-C finished!!!!!!!!!!!!!!!!!!!!!!!
done
~                                                   
