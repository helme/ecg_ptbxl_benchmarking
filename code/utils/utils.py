import os
import sys
import re
import glob
import pickle
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
import ast
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, roc_curve, auc
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import warnings

# EVALUATION STUFF
def generate_results(idxs, y_true, y_pred, thresholds):
    return evaluate_experiment(y_true[idxs], y_pred[idxs], thresholds)

def evaluate_experiment(y_true, y_pred, thresholds=None):
    results = {}

    if not thresholds is None:
        # binary predictions
        y_pred_binary = apply_thresholds(y_pred, thresholds)
        # PhysioNet/CinC Challenges metrics
        challenge_scores = challenge_metrics(y_true, y_pred_binary, beta1=2, beta2=2)
        results['F_beta_macro'] = challenge_scores['F_beta_macro']
        results['G_beta_macro'] = challenge_scores['G_beta_macro']

    # label based metric
    results['macro_auc'] = roc_auc_score(y_true, y_pred, average='macro')
    
    df_result = pd.DataFrame(results, index=[0])
    return df_result

def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=False):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:,classi], y_pred[:,classi]
        TP, FP, TN, FN = 0.,0.,0.,0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i]==y_predi[i]==1: 
                TP += 1./sample_weight
            if ((y_predi[i]==1) and (y_truei[i]!=y_predi[i])): 
                FP += 1./sample_weight
            if y_truei[i]==y_predi[i]==0: 
                TN += 1./sample_weight
            if ((y_predi[i]==0) and (y_truei[i]!=y_predi[i])): 
                FN += 1./sample_weight 
        f_beta_i = ((1+beta1**2)*TP)/((1+beta1**2)*TP + FP + (beta1**2)*FN)
        g_beta_i = (TP)/(TP+FP+beta2*FN)

        f_beta += f_beta_i
        g_beta += g_beta_i

    return {'F_beta_macro':f_beta/y_true.shape[1], 'G_beta_macro':g_beta/y_true.shape[1]}

def get_appropriate_bootstrap_samples(y_true, n_bootstraping_samples):
    samples=[]
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstraping_samples:
                break
    return samples

def find_optimal_cutoff_threshold(target, predicted):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = threshold[optimal_idx]
    return optimal_threshold

def find_optimal_cutoff_thresholds(y_true, y_pred):
	return [find_optimal_cutoff_threshold(y_true[:,i], y_pred[:,i]) for i in range(y_true.shape[1])]

def find_optimal_cutoff_threshold_for_Gbeta(target, predicted, n_thresholds=100):
    thresholds = np.linspace(0.00,1,n_thresholds)
    scores = [challenge_metrics(target, predicted>t, single=True)['G_beta_macro'] for t in thresholds]
    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx]

def find_optimal_cutoff_thresholds_for_Gbeta(y_true, y_pred):
    print("optimize thresholds with respect to G_beta")
    return [find_optimal_cutoff_threshold_for_Gbeta(y_true[:,k][:,np.newaxis], y_pred[:,k][:,np.newaxis]) for k in tqdm(range(y_true.shape[1]))]

def apply_thresholds(preds, thresholds):
	"""
		apply class-wise thresholds to prediction score in order to get binary format.
		BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
	"""
	tmp = []
	for p in preds:
		tmp_p = (p > thresholds).astype(int)
		if np.sum(tmp_p) == 0:
			tmp_p[np.argmax(p)] = 1
		tmp.append(tmp_p)
	tmp = np.array(tmp)
	return tmp

# DATA PROCESSING STUFF

def load_dataset(path, sampling_rate, release=False):
    if path.split('/')[-2] == 'ptbxl':
        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    elif path.split('/')[-2] == 'ICBEB':
        # load and convert annotation data
        Y = pd.read_csv(path+'icbeb_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_icbeb(Y, sampling_rate, path)

    return X, Y


def load_raw_data_icbeb(df, sampling_rate, path):

    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb

def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data
    with open(outputfolder+'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


# DOCUMENTATION STUFF

def generate_ptbxl_summary_table(selection=None, folder='../output/'):

    exps = ['exp0', 'exp1', 'exp1.1', 'exp1.1.1', 'exp2', 'exp3']
    metric1 = 'macro_auc'

    # get models
    models = {}
    for i, exp in enumerate(exps):
        if selection is None:
            exp_models = [m.split('/')[-1] for m in glob.glob(folder+str(exp)+'/models/*')]
        else:
            exp_models = selection
        if i == 0:
            models = set(exp_models)
        else:
            models = models.union(set(exp_models))

    results_dic = {'Method':[], 
                'exp0_AUC':[], 
                'exp1_AUC':[], 
                'exp1.1_AUC':[], 
                'exp1.1.1_AUC':[], 
                'exp2_AUC':[],
                'exp3_AUC':[]
                }

    for m in models:
        results_dic['Method'].append(m)
        
        for e in exps:
            
            try:
                me_res = pd.read_csv(folder+str(e)+'/models/'+str(m)+'/results/te_results.csv', index_col=0)
    
                mean1 = me_res.loc['point'][metric1]
                unc1 = max(me_res.loc['upper'][metric1]-me_res.loc['point'][metric1], me_res.loc['point'][metric1]-me_res.loc['lower'][metric1])

                results_dic[e+'_AUC'].append("%.3f(%.2d)" %(np.round(mean1,3), int(unc1*1000)))

            except FileNotFoundError:
                results_dic[e+'_AUC'].append("--")
            
            
    df = pd.DataFrame(results_dic)
    df_index = df[df.Method.isin(['naive', 'ensemble'])]
    df_rest = df[~df.Method.isin(['naive', 'ensemble'])]
    df = pd.concat([df_rest, df_index])
    df.to_csv(folder+'results_ptbxl.csv')

    titles = [
        '### 1. PTB-XL: all statements',
        '### 2. PTB-XL: diagnostic statements',
        '### 3. PTB-XL: Diagnostic subclasses',
        '### 4. PTB-XL: Diagnostic superclasses',
        '### 5. PTB-XL: Form statements',
        '### 6. PTB-XL: Rhythm statements'        
    ]

    # helper output function for markdown tables
    our_work = 'https://arxiv.org/abs/2004.13701'
    our_repo = 'https://github.com/helme/ecg_ptbxl_benchmarking/'
    md_source = ''
    for i, e in enumerate(exps):
        md_source += '\n '+titles[i]+' \n \n'
        md_source += '| Model | AUC &darr; | paper/source | code | \n'
        md_source += '|---:|:---|:---|:---| \n'
        for row in df_rest[['Method', e+'_AUC']].sort_values(e+'_AUC', ascending=False).values:
            md_source += '| ' + row[0].replace('fastai_', '') + ' | ' + row[1] + ' | [our work]('+our_work+') | [this repo]('+our_repo+')| \n'
    print(md_source)

def ICBEBE_table(selection=None, folder='../output/'):
    cols = ['macro_auc', 'F_beta_macro', 'G_beta_macro']

    if selection is None:
        models = [m.split('/')[-1].split('_pretrained')[0] for m in glob.glob(folder+'exp_ICBEB/models/*')]
    else:
        models = [] 
        for s in selection:
            #if s != 'Wavelet+NN':
                models.append(s)

    data = []
    for model in models:
        me_res = pd.read_csv(folder+'exp_ICBEB/models/'+model+'/results/te_results.csv', index_col=0)
        mcol=[]
        for col in cols:
            mean = me_res.ix['point'][col]
            unc = max(me_res.ix['upper'][col]-me_res.ix['point'][col], me_res.ix['point'][col]-me_res.ix['lower'][col])
            mcol.append("%.3f(%.2d)" %(np.round(mean,3), int(unc*1000)))
        data.append(mcol)
    data = np.array(data)

    df = pd.DataFrame(data, columns=cols, index=models)
    df.to_csv(folder+'results_icbeb.csv')

    df_rest = df[~df.index.isin(['naive', 'ensemble'])]
    df_rest = df_rest.sort_values('macro_auc', ascending=False)
    our_work = 'https://arxiv.org/abs/2004.13701'
    our_repo = 'https://github.com/helme/ecg_ptbxl_benchmarking/'

    md_source = '| Model | AUC &darr; |  F_beta=2 | G_beta=2 | paper/source | code | \n'
    md_source += '|---:|:---|:---|:---|:---|:---| \n'
    for i, row in enumerate(df_rest[cols].values):
        md_source += '| ' + df_rest.index[i].replace('fastai_', '') + ' | ' + row[0] + ' | ' + row[1] + ' | ' + row[2] + ' | [our work]('+our_work+') | [this repo]('+our_repo+')| \n'
    print(md_source)
