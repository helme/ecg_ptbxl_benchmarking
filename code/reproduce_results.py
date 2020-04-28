from experiments.scp_experiment import SCP_Experiment
from utils import utils
# model configs
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main():
    
    datafolder = '../data/ptbxl/'
    datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output/'

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
    # STANDARD SCP EXPERIMENTS ON PTBXL
    ##########################################

    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
       ]

    for name, task in experiments:
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()
        e.perform()
        e.evaluate()

    # generate greate summary table
    utils.generate_ptbxl_summary_table()

    ##########################################
    # EXPERIMENT BASED ICBEB DATA
    ##########################################

    e = SCP_Experiment('exp_ICBEB', 'all', datafolder_icbeb, outputfolder, models)
    e.prepare()
    e.perform()
    e.evaluate()

    # generate greate summary table
    utils.ICBEBE_table()

if __name__ == "__main__":
    main()
