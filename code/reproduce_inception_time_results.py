from experiments.scp_experiment import SCP_Experiment
from utils import utils
from code.configs.tf_inception_time_config import conf_tf_inception_all, conf_tf_inception_diagnostic, conf_tf_inception_form, conf_tf_inception_rhythm, conf_tf_inception_subdiagnostic, conf_tf_inception_superdiagnostic

def main():
    
    datafolder = '../data/ptbxl/'
    datafolder_icbeb = '../data/ICBEB/'
    outputfolder = '../output/'

    # perform each experiment one after another
    e = SCP_Experiment('custom_exp_name_1', 'all', datafolder, outputfolder, [conf_tf_inception_all])
    e.prepare()
    e.perform()
    e.evaluate()

    e = SCP_Experiment('custom_exp_name_2', 'diagnostic', datafolder, outputfolder, [conf_tf_inception_diagnostic])
    e.prepare()
    e.perform()
    e.evaluate()

    e = SCP_Experiment('custom_exp_name_3', 'subdiagnostic', datafolder, outputfolder, [conf_tf_inception_subdiagnostic])
    e.prepare()
    e.perform()
    e.evaluate()

    e = SCP_Experiment('custom_exp_name_4', 'superdiagnostic', datafolder, outputfolder, [conf_tf_inception_superdiagnostic])
    e.prepare()
    e.perform()
    e.evaluate()

    e = SCP_Experiment('custom_exp_name_5', 'form', datafolder, outputfolder, [conf_tf_inception_form])
    e.prepare()
    e.perform()
    e.evaluate()

    e = SCP_Experiment('custom_exp_name_6', 'rhythm', datafolder, outputfolder, [conf_tf_inception_rhythm])
    e.prepare()
    e.perform()
    e.evaluate()

if __name__ == "__main__":
    main()