from pytracking.evaluation.environment import EnvSettings
def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    # settings.got10k_path = '/home/dataset/GOT-10K-C/firstframe-corrupt/'
    settings.got10k_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/GOT-10K-C/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = ''
    settings.lasot_path = ''
    settings.network_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/pytracking/pytracking/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.oxuva_path = ''
    settings.result_plot_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/pytracking/pytracking/result_plots/'
    # settings.results_path = '/home/dataset/NIPS2022_workspace/GOT10K-C/firstframe-corrupt/'    # Where to store tracking results
    settings.results_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/workspace/GOT-10K-C/results'    # Where to store tracking results
    settings.segmentation_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/pytracking/pytracking/segmentation_results/'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = '/home/dataset/UAV123-C/'
    settings.vot_path = '/home/NIPS2022/workspace/vot2022/sequences/'
    settings.youtubevos_dir = ''
    settings.depthtrack_path = ''

    return settings    
