import numpy as np
import pandas as pd
import torch
import torch.utils.data
from torch import nn
from pathlib import Path
from scipy.stats import iqr
import os

#Note: due to issues with the numpy rng for multiprocessing (https://github.com/pytorch/pytorch/issues/5059) that could be fixed by a custom worker_init_fn we use random throught for convenience
import random

from skimage import transform

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

########################################################################
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz
#https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_filter(lowcut=10, highcut=20, fs=50, order=5, btype='band'):
    '''returns butterworth filter with given specifications'''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    sos = butter(order, [low, high] if btype=="band" else (low if btype=="low" else high), analog=False, btype=btype, output='sos')
    return sos

def butter_filter_frequency_response(filter):
    '''returns frequency response of a given filter (result of call of butter_filter)'''
    w, h = sosfreqz(filter)
    #gain vs. freq(Hz)
    #plt.plot((fs * 0.5 / np.pi) * w, abs(h))
    return w,h

def apply_butter_filter(data, filter, forwardbackward=True):
    '''pass filter from call of butter_filter to data (assuming time axis at dimension 0)'''
    if(forwardbackward):
        return sosfiltfilt(filter, data, axis=0)
    else:
        data = sosfilt(filter, data, axis=0)

######################################################################

def dataset_add_chunk_col(df, col="data"):
    '''add a chunk column to the dataset df'''
    df["chunk"]=df.groupby(col).cumcount()
    
def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col+"_length"]=df[col].apply(lambda x: len(np.load(x if data_folder is None else data_folder/x)))

def dataset_add_labels_col(df, col="label", data_folder=None):
    '''add a column with unique labels in column col'''
    df[col+"_labels"]=df[col].apply(lambda x: list(np.unique(np.load(x if data_folder is None else data_folder/x))))

def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_mean"]=df[col].apply(lambda x: np.mean(np.load(x if data_folder is None else data_folder/x),axis=axis))

def dataset_add_median_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with median'''
    df[col+"_median"]=df[col].apply(lambda x: np.median(np.load(x if data_folder is None else data_folder/x),axis=axis))

def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_std"]=df[col].apply(lambda x: np.std(np.load(x if data_folder is None else data_folder/x),axis=axis))

def dataset_add_iqr_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_iqr"]=df[col].apply(lambda x: iqr(np.load(x if data_folder is None else data_folder/x),axis=axis))

def dataset_get_stats(df, col="data",median=False):
    '''creates weighted means and stds from mean, std and length cols of the df'''
    mean = np.average(np.stack(df[col+("_median" if median is True else "_mean")],axis=0),axis=0,weights=np.array(df[col+"_length"]))
    std = np.average(np.stack(df[col+("_iqr" if median is True else "_std")],axis=0),axis=0,weights=np.array(df[col+"_length"]))
    return mean, std

def npys_to_memmap(npys, target_filename, delete_npys=False):
    memmap = None
    start = []
    length = []
    files= []
    ids=[]

    for idx,npy in enumerate(npys):
        data = np.load(npy)
        if(memmap is None):
            memmap = np.memmap(target_filename, dtype=data.dtype, mode='w+', shape=data.shape)
            start.append(0)
            length.append(data.shape[0])
        else:
            start.append(start[-1]+length[-1])
            length.append(data.shape[0])
            memmap = np.memmap(target_filename, dtype=data.dtype, mode='r+', shape=tuple([start[-1]+length[-1]]+[l for l in data.shape[1:]]))
        
        ids.append(idx)
        memmap[start[-1]:start[-1]+length[-1]]=data[:]
        memmap.flush()
        if(delete_npys is True):
            npy.unlink()
    del memmap

    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=[start[-1]+length[-1]]+[l for l in data.shape[1:]],dtype=data.dtype)

def reformat_as_memmap(df, target_filename, data_folder=None, annotation=False, delete_npys=False):
    npys_data = []
    npys_label = []
    
    for id,row in df.iterrows():
        npys_data.append(data_folder/row["data"] if data_folder is not None else row["data"])
        if(annotation):
            npys_label.append(data_folder/row["label"] if data_folder is not None else row["label"])

    npys_to_memmap(npys_data, target_filename, delete_npys=delete_npys)
    if(annotation):
        npys_to_memmap(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), delete_npys=delete_npys)
    
    #replace data(filename) by integer
    df_mapped = df.copy()
    df_mapped["data_original"]=df_mapped.data
    df_mapped["data"]=np.arange(len(df_mapped))
    df_mapped.to_pickle(target_filename.parent/("df_"+target_filename.stem+".pkl"))
    return df_mapped


##########################################################################
#TimeseriesDatasetCrops
##########################################################################
class TimeseriesDatasetCrops(torch.utils.data.Dataset):
    """timeseries dataset with partial crops."""

    def __init__(self, df, output_size, chunk_length, min_chunk_length, memmap_filename=None, npy_data=None, random_crop=True, data_folder=None, num_classes=2, copies=0, col_lbl="label", stride=None, start_idx=0, annotation=False, transforms=[]):
        """
        accepts three kinds of input:
        1) filenames pointing to aligned numpy arrays [timesteps,channels,...] for data and either integer labels or filename pointing to numpy arrays[timesteps,...] e.g. for annotations
        2) memmap_filename to memmap for data [concatenated,...] and labels- label column in df corresponds to index in this memmap
        3) npy_data [samples,ts,...] (either path or np.array directly- also supporting variable length input) - label column in df corresponds to sampleid
        
        transforms: list of callables (transformations) (applied in the specified order i.e. leftmost element first)
        """
        assert not((memmap_filename is not None) and (npy_data is not None))
        #require integer entries if using memmap or npy
        assert (memmap_filename is None and npy_data is None) or df.data.dtype==np.int64
                        
        self.timeseries_df = df
        self.output_size = output_size
        self.data_folder = data_folder
        self.transforms = transforms
        self.annotation = annotation
        self.col_lbl = col_lbl

        self.c = num_classes

        self.mode="files"
        self.memmap_filename = memmap_filename
        if(memmap_filename is not None):
            self.mode="memmap"
            memmap_meta = np.load(memmap_filename.parent/(memmap_filename.stem+"_meta.npz"))
            self.memmap_start = memmap_meta["start"]
            self.memmap_shape = tuple(memmap_meta["shape"])
            self.memmap_length = memmap_meta["length"]
            self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
            self.memmap_file_process_dict = {}
            if(annotation):
                memmap_meta_label = np.load(memmap_filename.parent/(memmap_filename.stem+"_label_meta.npz"))
                self.memmap_filename_label = memmap_filename.parent/(memmap_filename.stem+"_label.npy")
                self.memmap_shape_label = tuple(memmap_meta_label["shape"])
                self.memmap_file_process_dict_label = {}
                self.memmap_dtype_label = np.dtype(str(memmap_meta_label["dtype"]))
        elif(npy_data is not None):
            self.mode="npy"
            if(isinstance(npy_data,np.ndarray) or isinstance(npy_data,list)):
                self.npy_data = np.array(npy_data)
                assert(annotation is False)
            else:
                self.npy_data = np.load(npy_data)
            if(annotation):
                self.npy_data_label = np.load(npy_data.parent/(npy_data.stem+"_label.npy"))
        
        self.random_crop = random_crop

        self.df_idx_mapping=[]
        self.start_idx_mapping=[]
        self.end_idx_mapping=[]

        for df_idx,(id,row) in enumerate(df.iterrows()):
            if(self.mode=="files"):
                data_length = row["data_length"]
            elif(self.mode=="memmap"):
                data_length= self.memmap_length[row["data"]]
            else: #npy 
                data_length = len(self.npy_data[row["data"]])
                                              
            if(chunk_length == 0):#do not split
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx,data_length,chunk_length if stride is None else stride))
                idx_end = [min(l+chunk_length, data_length) for l in idx_start]

            #remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if(idx_end[i]-idx_start[i]< min_chunk_length):
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            #append to lists
            for _ in range(copies+1):
                for i_s,i_e in zip(idx_start,idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
                    
    def __len__(self):
        return len(self.df_idx_mapping)

    def __getitem__(self, idx):
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        #determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.output_size)
        if(self.random_crop):#random crop
            if(timesteps==self.output_size):
                start_idx_crop= start_idx
            else:
                start_idx_crop = start_idx + random.randint(0, timesteps - self.output_size -1)#np.random.randint(0, timesteps - self.output_size)
        else:
            start_idx_crop = start_idx + (timesteps - self.output_size)//2
        end_idx_crop = start_idx_crop+self.output_size

        #print(idx,start_idx,end_idx,start_idx_crop,end_idx_crop)
        #load the actual data
        if(self.mode=="files"):#from separate files
            data_filename = self.timeseries_df.iloc[df_idx]["data"]
            if self.data_folder is not None:
                data_filename = self.data_folder/data_filename
            data = np.load(data_filename)[start_idx_crop:end_idx_crop] #data type has to be adjusted when saving to npy
            
            ID = data_filename.stem

            if(self.annotation is True):
                label_filename = self.timeseries_df.iloc[df_idx][self.col_lbl]
                if self.data_folder is not None:
                    label_filename = self.data_folder/label_filename
                label = np.load(label_filename)[start_idx_crop:end_idx_crop] #data type has to be adjusted when saving to npy
            else:
                label = self.timeseries_df.iloc[df_idx][self.col_lbl] #input type has to be adjusted in the dataframe
        elif(self.mode=="memmap"): #from one memmap file
            ID = self.timeseries_df.iloc[df_idx]["data_original"].stem
            memmap_idx = self.timeseries_df.iloc[df_idx]["data"] #grab the actual index (Note the df to create the ds might be a subset of the original df used to create the memmap)
            idx_offset = self.memmap_start[memmap_idx]
            
            pid = os.getpid()
            #print("idx",idx,"ID",ID,"idx_offset",idx_offset,"start_idx_crop",start_idx_crop,"df_idx", self.df_idx_mapping[idx],"pid",pid)
            mem_file = self.memmap_file_process_dict.get(pid, None)  # each process owns its handler.
            if mem_file is None:
                #print("memmap_shape", self.memmap_shape)
                mem_file = np.memmap(self.memmap_filename, self.memmap_dtype, mode='r', shape=self.memmap_shape)
                self.memmap_file_process_dict[pid] = mem_file
            data = np.copy(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            #print(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            if(self.annotation):
                mem_file_label = self.memmap_file_process_dict_label.get(pid, None)  # each process owns its handler.
                if mem_file_label is None:
                    mem_file_label = np.memmap(self.memmap_filename_label, self.memmap_dtype, mode='r', shape=self.memmap_shape_label)
                    self.memmap_file_process_dict_label[pid] = mem_file_label
                label = np.copy(mem_file_label[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            else:
                label = self.timeseries_df.iloc[df_idx][self.col_lbl]
        else:#single npy array
            ID = self.timeseries_df.iloc[df_idx]["data"]
            
            data = self.npy_data[ID][start_idx_crop:end_idx_crop]
            
            if(self.annotation):
                label = self.npy_data_label[ID][start_idx_crop:end_idx_crop]
            else:
                label = self.timeseries_df.iloc[df_idx][self.col_lbl]
        sample = {'data': data, 'label': label, 'ID':ID}
        
        for t in self.transforms:
            sample = t(sample)

        return sample
    
    def get_sampling_weights(self, class_weight_dict,length_weighting=False, group_by_col=None):
        assert(self.annotation is False)
        assert(length_weighting is False or group_by_col is None)
        weights = np.zeros(len(self.df_idx_mapping),dtype=np.float32)
        length_per_class = {}
        length_per_group = {}
        for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
            label = self.timeseries_df.iloc[i][self.col_lbl]
            weight = class_weight_dict[label]
            if(length_weighting):
                if label in length_per_class.keys():
                    length_per_class[label] += e-s
                else:
                    length_per_class[label] = e-s
            if(group_by_col is not None):
                group = self.timeseries_df.iloc[i][group_by_col]
                if group in length_per_group.keys():
                    length_per_group[group] += e-s
                else:
                    length_per_group[group] = e-s
            weights[iw] = weight

        if(length_weighting):#need second pass to properly take into account the total length per class
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                label = self.timeseries_df.iloc[i][self.col_lbl]
                weights[iw]= (e-s)/length_per_class[label]*weights[iw]
        if(group_by_col is not None):
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                group = self.timeseries_df.iloc[i][group_by_col]
                weights[iw]= (e-s)/length_per_group[group]*weights[iw]

        weights = weights/np.min(weights)#normalize smallest weight to 1
        return weights

    def get_id_mapping(self):
        return self.df_idx_mapping

class RandomCrop(object):
    """Crop randomly the image in a sample (deprecated).
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, ID = sample['data'], sample['label'], sample['ID']

        
        timesteps= len(data)
        assert(timesteps>=self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)

        data = data[start: start + self.output_size]
        if(self.annotation):
            label = label[start: start + self.output_size]

        return {'data': data, 'label': label, "ID":ID}

class CenterCrop(object):
    """Center crop the image in a sample (deprecated).
    """

    def __init__(self, output_size, annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, ID = sample['data'], sample['label'], sample['ID']

        
        timesteps= len(data)

        start = (timesteps - self.output_size)//2

        data = data[start: start + self.output_size]
        if(self.annotation):
            label = label[start: start + self.output_size]

        return {'data': data, 'label': label, "ID":ID}

class GaussianNoise(object):
    """Add gaussian noise to sample.
    """

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, sample):
        if self.scale ==0:
            return sample
        else:
            data, label, ID = sample['data'], sample['label'], sample['ID']
            data = data + np.reshape(np.array([random.gauss(0,self.scale) for _ in range(np.prod(data.shape))]),data.shape)#np.random.normal(scale=self.scale,size=data.shape).astype(np.float32)
            return {'data': data, 'label': label, "ID":ID}
        
class Rescale(object):
    """Rescale by factor.
    """

    def __init__(self, scale=0.5,interpolation_order=3):
        self.scale = scale
        self.interpolation_order = interpolation_order

    def __call__(self, sample):
        if self.scale ==1:
            return sample
        else:
            data, label, ID = sample['data'], sample['label'], sample['ID']
            timesteps_new = int(self.scale * len(data))
            data = transform.resize(data,(timesteps_new,data.shape[1]),order=interpolation_order).astype(np.float32)
            return {'data': data, 'label': label, "ID":ID}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, transpose_data1d=True):
        self.transpose_data1d=transpose_data1d

    def __call__(self, sample):
        def _to_tensor(data,transpose_data1d=False):
            if(len(data.shape)==2 and transpose_data1d is True):#swap channel and time axis for direct application of pytorch's 1d convs
                data = data.transpose((1,0))
            if(isinstance(data,np.ndarray)):
                return torch.from_numpy(data)
            else:#default_collate will take care of it
                return data
            
        data, label, ID = sample['data'], sample['label'], sample['ID']

        if not isinstance(data,tuple): 
            data = _to_tensor(data,self.transpose_data1d)
        else:
            data = tuple(_to_tensor(x,self.transpose_data1d) for x in data)
        
        if not isinstance(label,tuple):
            label = _to_tensor(label)
        else:
            label = tuple(_to_tensor(x) for x in label)

        return data,label #returning as a tuple (potentially of lists)


class Normalize(object):
    """Normalize using given stats.
    """

    def __init__(self, stats_mean, stats_std, input=True, channels=[]):
        self.stats_mean=np.expand_dims(stats_mean.astype(np.float32),axis=0) if stats_mean is not None else None
        self.stats_std=np.expand_dims(stats_std.astype(np.float32),axis=0)+1e-8 if stats_std is not None else None
        self.input = input
        if(len(channels)>0):
            for i in range(len(stats_mean)):
                if(not(i in channels)):
                    self.stats_mean[:,i]=0
                    self.stats_std[:,i]=1

    def __call__(self, sample):
        if(self.input):
            data = sample['data']
        else:
            data = sample['label']


        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if(self.stats_std is not None):
            data = data/self.stats_std
        
        if(self.input):
            return {'data': data, 'label': sample['label'], "ID":sample['ID']}
        else:
            return {'data': sample['data'], 'label': data, "ID":sample['ID']}             

class ButterFilter(object):
    """Normalize using given stats.
    """

    def __init__(self, lowcut=50, highcut=50, fs=100, order=5, btype='band', forwardbackward=True, input=True):
        self.filter = butter_filter(lowcut,highcut,fs,order,btype)
        self.input = input
        self.forwardbackward = forwardbackward

    def __call__(self, sample):
        if(self.input):
            data = sample['data']
        else:
            data = sample['label']

        #check multiple axis
        if(self.forwardbackward):
            data = sosfiltfilt(self.filter, data, axis=0)
        else:
            data = sosfilt(self.filter, data, axis=0)
    
        if(self.input):
            return {'data': data, 'label': sample['label'], "ID":sample['ID']}
        else:
            return {'data': sample['data'], 'label': data, "ID":sample['ID']}

class ChannelFilter(object):
    """Select certain channels.
    """

    def __init__(self, channels=[0], input=True):
        self.channels = channels
        self.input = input

    def __call__(self, sample):
        if(self.input):
            return {'data': sample['data'][:,self.channels], 'label': sample['label'], "ID":sample['ID']}
        else:
            return {'data': sample['data'], 'label': sample['label'][:,self.channels], "ID":sample['ID']}

class Transform(object):
    """Transforms data using a given function i.e. data_new = func(data) for input is True else label_new = func(label)
    """

    def __init__(self, func, input=False):
        self.func = func
        self.input = input

    def __call__(self, sample):
        if(self.input):
            return {'data': self.func(sample['data']), 'label': sample['label'], "ID":sample['ID']}
        else:
            return {'data': sample['data'], 'label': self.func(sample['label']), "ID":sample['ID']}

class TupleTransform(object):
    """Transforms data using a given function (operating on both data and label and return a tuple) i.e. data_new, label_new = func(data_old, label_old)
    """

    def __init__(self, func, input=False):
        self.func = func

    def __call__(self, sample):
        data_new, label_new = self.func(sample['data'],sample['label'])
        return {'data': data_new, 'label': label_new, "ID":sample['ID']}
        
##########################################################################
#MIL and ensemble models
##########################################################################
def aggregate_predictions(preds,targs=None,idmap=None,aggregate_fn = np.mean,verbose=True):
    '''
    aggregates potentially multiple predictions per sample (can also pass targs for convenience)
    idmap: idmap as returned by TimeSeriesCropsDataset's get_id_mapping
    preds: ordered predictions as returned by learn.get_preds()
    aggregate_fn: function that is used to aggregate multiple predictions per sample (most commonly np.amax or np.mean)
    '''
    if(idmap is not None and len(idmap)!=len(np.unique(idmap))):
        if(verbose):
            print("aggregating predictions...")
            preds_aggregated = []
            targs_aggregated = []
            for i in np.unique(idmap):
                preds_local = preds[np.where(idmap==i)[0]]
                preds_aggregated.append(aggregate_fn(preds_local,axis=0))
                if targs is not None:
                    targs_local = targs[np.where(idmap==i)[0]]
                    assert(np.all(targs_local==targs_local[0])) #all labels have to agree
                    targs_aggregated.append(targs_local[0])
            if(targs is None):
                return np.array(preds_aggregated)
            else:
                return np.array(preds_aggregated),np.array(targs_aggregated)
    else:
        if(targs is None):
            return preds
        else:
            return preds,targs
    
    
class milwrapper(nn.Module):
    def __init__(self,model,input_size,n,stride=None,softmax=True):
        super().__init__()
        self.n = n
        self.input_size = input_size
        self.model = model
        self.softmax = softmax
        self.stride = input_size if stride is None else stride
        
    def forward(self,x):
        #bs,ch,seq
        for i in range(self.n):
            pred_single = self.model(x[:,:,i*self.stride:i*self.stride+self.input_size])
            pred_single = nn.functional.softmax(pred_single,dim=1)
            if(i==0):
                pred= pred_single
            else:
                pred += pred_single
        return pred/self.n

class ensemblewrapper(nn.Module):
    def __init__(self,model,checkpts):
        super().__init__()
        self.model = model
        self.checkpts = checkpts
        
        
    def forward(self,x):
        #bs,ch,seq
        for i,c in enumerate(self.checkpts):
            state = torch.load(Path("./models/")/f'{c}.pth', map_location=x.device)
            self.model.load_state_dict(state['model'], strict=True)

            pred_single = self.model(x)
            pred_single = nn.functional.softmax(pred_single,dim=1)
            if(i==0):
                pred= pred_single
            else:
                pred += pred_single
        return pred/len(self.checkpts)
