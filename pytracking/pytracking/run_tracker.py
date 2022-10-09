import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import get_dataset
from pytracking.evaluation.running import run_dataset
from pytracking.evaluation import Tracker
import cProfile
# import re
# cProfile.run('re.compile("foo|bar")')

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                visdom_info=None,result_name = 'dimp', blend=0,visual=0, corrupt=0, datasetpath=None ):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
        visdom_info: Dict optionally containing 'use_visdom', 'server' and 'port' for Visdom visualization.
    """

    visdom_info = {} if visdom_info is None else visdom_info

    dataset = get_dataset(dataset_name, datasetpath=datasetpath)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, run_id, resultname=result_name)]

    run_dataset(dataset, trackers, debug, threads, visdom_info=visdom_info,blend=blend, visual=visual, corrupt=corrupt)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom.')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom.')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom.')
    parser.add_argument('--gpuid', type=str, default='0', help='gpu id.')
    parser.add_argument('--result_name', type=str, default='', help='result_name.')
    parser.add_argument('--blend', type=float, default=0, help='blend_parameter')
    parser.add_argument('--visual', type=int, default=0, help='visual')
    parser.add_argument('--corrupt', type=int, default=0, help='random corruption, default 0 no corruption')
    parser.add_argument('--datasetpath', type=str, default=None, help='set your datasetpath without local.py')
    
    parser.add_argument('--corp_dataset', type=str, default="gaussian", help='corp_dataset')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}, result_name = args.result_name, blend=args.blend, visual=args.visual, corrupt=args.corrupt, datasetpath=args.datasetpath)


if __name__ == '__main__':
    main()
