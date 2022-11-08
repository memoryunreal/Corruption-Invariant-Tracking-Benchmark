from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.avist_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/avist'
    settings.got10k_lmdb_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/got10k_lmdb'
    settings.got10k_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/lasot_lmdb'
    settings.lasot_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/lasot'
    settings.network_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/nfs'
    settings.otb_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/OTB2015'
    settings.prj_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark'
    settings.result_plot_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/test/result_plots'
    settings.results_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark'
    settings.segmentation_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/test/segmentation_results'
    settings.tc128_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/trackingNet'
    settings.uav_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/UAV123'
    settings.vot_path = '/home/CVPR2023/Corruption-Invariant-Tracking-Benchmark/other_tracker/Stark/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

