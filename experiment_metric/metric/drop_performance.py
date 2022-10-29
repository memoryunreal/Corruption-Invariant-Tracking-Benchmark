import os
corp_dataset_srao=[
"Trackers: dimp_dimp18                    success50: 77.8        AO: 66.6",
"Trackers: dimp_dimp50                    success50: 81.3        AO: 69.6",
"Trackers: dimp_prdimp50                  success50: 81.6        AO: 70.7",
"Trackers: dimp_prdimp18                  success50: 81.4        AO: 70.5",
"Trackers: keep_track_default             success50: 85.4        AO: 74.5",
"Trackers: keep_track_default_fast        success50: 86.8        AO: 75.3",
"Trackers: kys_default                    success50: 81.9        AO: 69.8",
"Trackers: mixformer1k                    success50: 89.3        AO: 79.5",
"Trackers: mixformer1k-got                success50: 85.6        AO: 76.8",
"Trackers: mixformer22k                   success50: 92.1        AO: 81.9",
"Trackers: mixformer22k-got               success50: 89.0        AO: 79.4",
"Trackers: mixformerL                     success50: 90.9        AO: 81.3",
"Trackers: tomp_tomp101                   success50: 88.7        AO: 77.6",
"Trackers: tomp_tomp50                    success50: 89.9        AO: 78.1",
]
clean_dataset_srao=[ 
"Trackers: dimp_dimp18                    success50: 86.1        AO: 73.0",
"Trackers: dimp_dimp50                    success50: 88.9        AO: 75.4",
"Trackers: dimp_prdimp18                  success50: 87.5        AO: 75.8",
"Trackers: dimp_prdimp50                  success50: 89.4        AO: 77.7",
"Trackers: keep_track_default             success50: 92.2        AO: 80.8",
"Trackers: keep_track_default_fast        success50: 92.1        AO: 80.5",
"Trackers: kys_default                    success50: 89.2        AO: 75.6",
"Trackers: mixformer1k-got                success50: 94.7        AO: 85.7",
"Trackers: mixformer1k                    success50: 95.4        AO: 86.1",
"Trackers: mixformer22k                   success50: 94.6        AO: 85.0",
"Trackers: mixformer22k-got               success50: 94.8        AO: 85.2",
"Trackers: mixformerL                     success50: 94.8        AO: 85.8",
"Trackers: tomp_tomp101                   success50: 93.6        AO: 83.0",
"Trackers: tomp_tomp50                    success50: 94.7        AO: 83.9",
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
    for i in range(len(clean_dataset)):
        clean_sr = float(clean_dataset[i][52:57]) 
        clean_ao = float(clean_dataset[i][68:74]) 
        corp_sr = float(corp_dataset[i][52:57])
        corp_ao = float(corp_dataset[i][68:74]) 
        trackername = clean_dataset[i][10:40].split(" ")[0] # tracker name

        drop_sr = (clean_sr - corp_sr) / clean_sr * 100
        drop_ao = (clean_ao - corp_ao) / clean_ao * 100
        print('{:<30} & {:<7.1f} & {:<7.1f} & {:.1f}({:<6.2f}) & {:.1f}({:<6.2f})'.format(trackername, clean_sr, clean_ao, corp_sr, drop_sr, corp_ao, drop_ao))

if __name__ == "__main__":
    # drop_sr_ao(clean_dataset_srao, corp_dataset_srao)
    drop_votlt(clean_votlt, corp_votlt)
