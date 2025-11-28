"""
Reproduce results using only Lead II ECG data

This script demonstrates how to run experiments using only Lead II instead of all 12 leads.
Simply pass leads='II' parameter to the SCP_Experiment constructor.
"""

import argparse
import os
from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main(device='cuda:0'):
    # Set CUDA device
    if device.startswith('cuda:'):
        device_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = device_id
        print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={device_id})")
    else:
        print(f"Using device: {device}")

    datafolder = '../dataset/1.0.1/ptbxl/'
    # datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output_leadII/'

    models = [
        conf_fastai_xresnet1d101,
        conf_fastai_resnet1d_wang,
        conf_fastai_lstm,
        conf_fastai_lstm_bidir,
        conf_fastai_fcn_wang,
        conf_fastai_inception1d,
        conf_wavelet_standard_nn,
        ]

    ##########################################
    # LEAD II ONLY EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('exp0_lead_ii', 'all'),
        ('exp1_lead_ii', 'diagnostic'),
        ('exp1.1_lead_ii', 'subdiagnostic'),
        ('exp1.1.1_lead_ii', 'superdiagnostic'),
        ('exp2_lead_ii', 'form'),
        ('exp3_lead_ii', 'rhythm')
       ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models, leads='II')
        e.prepare()
        e.perform()
        e.evaluate()

    # Note: You can also use a subset of leads by passing a list
    # For example: leads=['I', 'II', 'V1'] for only those three leads

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reproduce results using only Lead II ECG data')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training (e.g., cuda:0, cuda:1, cpu). Default: cuda:0')
    args = parser.parse_args()

    main(device=args.device)
