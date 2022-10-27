import os
corp_dataset=[
"Trackers: DiMP50                         success50: 56.88       AO: 48.29",
"Trackers: PrDiMP18                       success50: 53.96       AO: 45.52",
"Trackers: PrDiMP50                       success50: 54.92       AO: 47.01",
"Trackers: DiMP18                         success50: 59.86       AO: 49.23",
"Trackers: ToMP101                        success50: 72.00       AO: 59.06",
"Trackers: ToMP50                         success50: 71.02       AO: 58.91",
"Trackers: kys                            success50: 54.42       AO: 46.15",
"Trackers: keeptrack                      success50: 63.03       AO: 52.43",
]
clean_dataset=[
"Trackers: DiMP18                         success50: 63.80       AO: 52.36",
"Trackers: DiMP50                         success50: 71.15       AO: 58.84",
"Trackers: PrDiMP18                       success50: 65.06       AO: 53.70",
"Trackers: ToMP101                        success50: 82.95       AO: 68.20",
"Trackers: ToMP50                         success50: 76.40       AO: 63.54",
"Trackers: keeptrack                      success50: 76.37       AO: 63.29",
"Trackers: PrDiMP50                       success50: 67.38       AO: 55.91",
"Trackers: kys                            success50: 63.89       AO: 53.80",
]

if __name__ == "__main__":
    for i in range(len(clean_dataset)):
        clean_sr = float(clean_dataset[i][52:57]) 
        clean_ao = float(clean_dataset[i][68:74]) 
        corp_sr = float(corp_dataset[i][52:57])
        corp_ao = float(corp_dataset[i][68:74]) 
        trackername = clean_dataset[i][10:40].split(" ")[0] # tracker name

        drop_sr = (clean_sr - corp_sr) / clean_sr * 100
        drop_ao = (clean_ao - corp_ao) / clean_ao * 100
        print('{:<30} & {:<7.2f} & {:<7.2f} & {:.2f}({:<6.2f}) & {:.2f}({:<6.2f})'.format(trackername, clean_sr, clean_ao, corp_sr, drop_sr, corp_ao, drop_ao))