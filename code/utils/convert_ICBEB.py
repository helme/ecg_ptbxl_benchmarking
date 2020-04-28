import os
import pandas as pd
import wfdb
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom
from scipy.io import loadmat
from stratisfy import stratisfy_df

output_folder = 'data/ICBEB/'
output_datafolder_100 = output_folder+ '/records100/'
output_datafolder_500 = output_folder+ '/records500/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(output_datafolder_100):
    os.makedirs(output_datafolder_100)
if not os.path.exists(output_datafolder_500):
    os.makedirs(output_datafolder_500)
    
def store_as_wfdb(signame, data, sigfolder, fs):
    channel_itos=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    wfdb.wrsamp(signame,
                fs=fs,
                sig_name=channel_itos, 
                p_signal=data,
                units=['mV']*len(channel_itos),
                fmt = ['16']*len(channel_itos), 
                write_dir=sigfolder)  

df_reference = pd.read_csv('tmp_data/REFERENCE.csv')

label_dict = {1:'NORM', 2:'AFIB', 3:'1AVB', 4:'CLBBB', 5:'CRBBB', 6:'PAC', 7:'VPC', 8:'STD_', 9:'STE_'}

data = {'ecg_id':[], 'filename':[], 'validation':[], 'age':[], 'sex':[], 'scp_codes':[]}

ecg_counter = 0
for folder in ['TrainingSet1', 'TrainingSet2', 'TrainingSet3']:
    filenames = os.listdir('tmp_data/'+folder)
    for filename in tqdm(filenames):
        if filename.split('.')[1] == 'mat':
            ecg_counter += 1
            name = filename.split('.')[0]
            sex, age, sig = loadmat('tmp_data/'+folder+'/'+filename)['ECG'][0][0]
            data['ecg_id'].append(ecg_counter)
            data['filename'].append(name)
            data['validation'].append(False)
            data['age'].append(age[0][0])
            data['sex'].append(1 if sex[0] == 'Male' else 0)
            labels = df_reference[df_reference.Recording == name][['First_label' ,'Second_label' ,'Third_label']].values.flatten()
            labels = labels[~np.isnan(labels)].astype(int)
            data['scp_codes'].append({label_dict[key]:100 for key in labels})
            store_as_wfdb(str(ecg_counter), sig.T, output_datafolder_500, 500)
            down_sig = np.array([zoom(channel, .2) for channel in sig])
            store_as_wfdb(str(ecg_counter), down_sig.T, output_datafolder_100, 100)

df = pd.DataFrame(data)
df['patient_id'] = df.ecg_id
df = stratisfy_df(df, 'strat_fold')
df.to_csv(output_folder+'icbeb_database.csv')