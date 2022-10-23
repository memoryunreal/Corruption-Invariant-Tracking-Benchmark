import numpy as np
origin = []

trackerlist = ['DeT', 'DAL','TSDM','CSR_RGBD++', 'DSKCF_shape']
normal_res = [0.588, 0.544, 0.484, 0.166, 0.040]
robust = ['RMN','RMA','RMM']
# rob_result= [[0.450,0.578,0.586],
# [0.466,0.543,0.538],
# [0.390,0.482,0.477],
# [0.460,0.604,0.609],
# [0.085,0.148,0.124],
# [0.037,0.034,0.039]]
sre = ["SRE", "TRE","RTRE"]
sre_result = [
[0.512, 0.578, 0.457],
[0.484, 0.544, 0.449],
[0.432, 0.474, 0.323],
[0.148, 0.165, 0.100],
[0.035, 0.0396, 0.0292 ]]

matrix_result = []
for id,tracker in enumerate(trackerlist):
    percentage = [(normal_res[id] - sre_result[id][index])/normal_res[id]  for index,rob in enumerate(sre)] 
    matrix_result.append([percentage[0]*100, percentage[1]*100, percentage[2]*100])
    print('tracker {} : {} {} {} '.format(tracker, percentage[0]*100, percentage[1]*100, percentage[2]*100))
# ATCAIS20    : 1.087277003249107
# DDiMP       : 3.1787498714079985
# iiau_rgbd   : 5.074597079707859
# SiamM_Ds    : 14.328292715500835
# sttc_rgbd   : 18.83953972805051
# TSDM        : 15.577865045723179
# DAL         : 28.845176679273457
# DRefine     : 4.098044802027083
# SiamDW_D    : 2.25484597887608
# SLMD        : 2.2642716956103786
# TALGD       : 1.3644797957048118

# DeT \cite{yan2021depthtrack}        &0.512  &0.578/ & 0.457/
# DAL \cite{2019DAL}                  &0.484  &0.544/ & 0.449/
# TSDM \cite{zhao2021tsdm}            &0.432  &0.474/ & 0.323/
# CSR\_RGBD++ \cite{kart2018depth}    &0.148  &0.166/ & 0.100/
# DS\_KCF\_shape \cite{2016DS}        &0.035  &0.0396 & 0.0292/ 
# \hline
# Average                             &0.3695/-8.29\% &0.4114/+2.11\% & 0.3230/-19.8\%\\
print(matrix_result)
matrix_result.append([7.76, -1.52, 11.7])
matrix =  np.array([[23.46, 1.700, 0.34],
            [14.33, 0.183, 1.10],
            [19.42, 0.413, 1.44],
            [24.59, 0.983, 0.16],
            [48.79, 10.84, 25.3],
            [7.500, 14.99, 2.50]])
np.average(matrix, axis=0)
print("average: {}".format(np.average(matrix_result, axis=0)))
print('hello')