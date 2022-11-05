import os
corp_dataset_srao=[
"Trackers: mixformer1k-got                success50: 85.56       AO: 76.78",
"Trackers: mixformer1k                    success50: 89.27       AO: 79.51",
"Trackers: mixformer22k                   success50: 92.12       AO: 81.89",
"Trackers: dimp_dimp50                    success50: 81.33       AO: 69.64",
"Trackers: dimp_prdimp18                  success50: 81.42       AO: 70.48",
"Trackers: kys_default                    success50: 81.93       AO: 69.84",
"Trackers: keep_track_default_fast        success50: 86.75       AO: 75.29",
"Trackers: dimp_prdimp50                  success50: 81.65       AO: 70.73",
"Trackers: keep_track_default             success50: 85.38       AO: 74.54",
"Trackers: dimp_dimp18                    success50: 77.81       AO: 66.57",
"Trackers: mixformerL                     success50: 90.92       AO: 81.28",
"Trackers: ostrack256                     success50: 89.87       AO: 79.85",
"Trackers: mixformer22k-got               success50: 89.00       AO: 79.38",
"Trackers: ostrack384                     success50: 90.86       AO: 81.36",
"Trackers: ostrack384-got                 success50: 89.05       AO: 79.66",
"Trackers: stark101                       success50: 87.00       AO: 76.70",
"Trackers: stark101-got                   success50: 84.80       AO: 75.01",
"Trackers: ostrack256-got                 success50: 88.84       AO: 79.02",
"Trackers: tomp_tomp101                   success50: 88.68       AO: 77.59",
"Trackers: tomp_tomp50                    success50: 89.90       AO: 78.10",
"Trackers: stark50                        success50: 86.37       AO: 76.33",
"Trackers: stark50-got                    success50: 83.86       AO: 74.38",
]
clean_dataset_srao=[ 
"Trackers: kys_default                    success50: 89.15       AO: 75.62",
"Trackers: dimp_prdimp18                  success50: 87.52       AO: 75.81",
"Trackers: mixformer1k-got                success50: 94.69       AO: 85.66",
"Trackers: dimp_dimp50                    success50: 88.91       AO: 75.38",
"Trackers: mixformer1k                    success50: 95.39       AO: 86.10",
"Trackers: keep_track_default             success50: 92.22       AO: 80.78",
"Trackers: keep_track_default_fast        success50: 92.05       AO: 80.52",
"Trackers: mixformer22k                   success50: 94.56       AO: 84.95",
"Trackers: dimp_dimp18                    success50: 86.10       AO: 73.00",
"Trackers: dimp_prdimp50                  success50: 89.41       AO: 77.73",
"Trackers: ostrack256                     success50: 95.66       AO: 86.20",
"Trackers: mixformer22k-got               success50: 94.81       AO: 85.21",
"Trackers: mixformerL                     success50: 94.83       AO: 85.84",
"Trackers: stark50-got                    success50: 93.14       AO: 83.34",
"Trackers: ostrack256-got                 success50: 95.76       AO: 86.22",
"Trackers: stark50                        success50: 93.42       AO: 83.22",
"Trackers: ostrack384-got                 success50: 96.15       AO: 86.78",
"Trackers: stark101                       success50: 93.31       AO: 83.54",
"Trackers: ostrack384                     success50: 96.23       AO: 87.20",
"Trackers: stark101-got                   success50: 92.07       AO: 82.40",
"Trackers: tomp_tomp101                   success50: 93.64       AO: 83.01",
"Trackers: tomp_tomp50                    success50: 94.71       AO: 83.87",
]

corp_votlt= [  
"DiMP18	0.224	0.442	0.628",
"DiMP50	0.234	0.438	0.667",
"PrDiMP18	0.219	0.449	0.616",
"PrDiMP50	0.229	0.448	0.641",
"ToMP101	0.250	0.433	0.715",
"ToMP50	0.247	0.429	0.708",
"keeptrack	0.239	0.465	0.661",
"kys	0.231	0.439	0.657",
"lwl	0.275	0.647	0.555",
"mixformer	0.412	0.684	0.769",
"mixformer1k	0.383	0.660	0.723",
"mixformerL	0.434	0.677	0.798",
]

clean_votlt = [
"DiMP18	0.275	0.451	0.747",
"DiMP50	0.271	0.455	0.732",
"PrDiMP18	0.262	0.468	0.706",
"PrDiMP50	0.280	0.475	0.747",
"ToMP101	0.298	0.454	0.792",
"ToMP50	0.301	0.451	0.799",
"keeptrack	0.294	0.478	0.773",
"kys	0.270	0.454	0.733",
"lwl	0.319	0.708	0.608",
"mixformer	0.533	0.760	0.851",
"mixformer1k	0.527	0.745	0.833",
"mixformerL	0.554	0.761	0.853",
]

def drop_votlt(clean_votlt, corp_votlt):
    for i in range(len(corp_votlt)):
        clean_str = clean_votlt[i].split()
        clean_pr = float(clean_str[1])*100
        clean_re = float(clean_str[2])*100
        clean_f = float(clean_str[3])*100

        corp_str = corp_votlt[i].split()
        corp_pr = float(corp_str[1])*100
        corp_re = float(corp_str[2])*100
        corp_f = float(corp_str[3])*100

        assert clean_str[0] == corp_str[0]

        drop_pr = (clean_pr - corp_pr) / clean_pr * 100
        drop_re = (clean_re - corp_re) / clean_re * 100
        drop_f = (clean_f - corp_f) / clean_f * 100

        print('{:<30} & {:<7.1f} & {:<7.1f} & {:<7.1f} & {:.1f}({:<6.2f}) & {:.1f}({:<6.2f}) & {:.1f}({:<6.2f})'.format(
            clean_str[0], clean_pr, clean_re, clean_f, corp_pr, drop_pr, corp_re, drop_re, corp_f, drop_f))

def drop_sr_ao(clean_dataset, corp_dataset):
    # read tracker
    clean_dict = {}
    corp_dict = {}
    for i in range(len(clean_dataset)):
        dict1 = {}
        dict2 = {} 
        clean_sr = float(clean_dataset[i][52:57]) 
        clean_ao = float(clean_dataset[i][68:74]) 
        corp_sr = float(corp_dataset[i][52:57])
        corp_ao = float(corp_dataset[i][68:74]) 
        trackername = clean_dataset[i][10:40].split(" ")[0] # tracker name
        trackername_corp = corp_dataset[i][10:40].split(" ")[0] # tracker name

        dict1["sr"] = clean_sr
        dict1["ao"] = clean_ao
        dict2["sr"] = corp_sr
        dict2["ao"] = corp_ao
        clean_dict[trackername]= dict1
        corp_dict[trackername_corp] = dict2

    for k in clean_dict:
        clean_sr_r = clean_dict[k]['sr']
        corp_sr_r = corp_dict[k]['sr']
        clean_ao_r = clean_dict[k]['ao']
        corp_ao_r = corp_dict[k]['ao']
        drop_sr = (clean_sr_r - corp_sr_r) / clean_sr_r * 100
        drop_ao = (clean_ao_r - corp_ao_r) / clean_ao_r * 100
        print('{:<30} & {:<7.1f} & {:<7.1f} & {:.1f}({:<6.2f}) & {:.1f}({:<6.2f})'.format(k, clean_sr_r, clean_ao_r, corp_sr_r, drop_sr, corp_ao_r, drop_ao))

if __name__ == "__main__":
    drop_sr_ao(clean_dataset_srao, corp_dataset_srao)
    # drop_votlt(clean_votlt, corp_votlt)
