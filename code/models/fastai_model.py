from models.timeseries_utils import *

from fastai import *
from fastai.basic_data import *
from fastai.basic_train import *
from fastai.train import *
from fastai.metrics import *
from fastai.torch_core import *
from fastai.callbacks.tracker import SaveModelCallback

from pathlib import Path
from functools import partial

from models.resnet1d import resnet1d18,resnet1d34,resnet1d50,resnet1d101,resnet1d152,resnet1d_wang,resnet1d,wrn1d_22
from models.xresnet1d import xresnet1d18,xresnet1d34,xresnet1d50,xresnet1d101,xresnet1d152,xresnet1d18_deep,xresnet1d34_deep,xresnet1d50_deep,xresnet1d18_deeper,xresnet1d34_deeper,xresnet1d50_deeper
from models.inception1d import inception1d
from models.basic_conv1d import fcn,fcn_wang,schirrmeister,sen,basic1d,weight_init
from models.rnn1d import RNN1d
import math

from models.base_model import ClassificationModel
import torch 

#for lrfind
import matplotlib
import matplotlib.pyplot as plt

#eval for early stopping
from fastai.callback import Callback
from utils.utils import evaluate_experiment

class metric_func(Callback):
    "Obtains score using user-supplied function func (potentially ignoring targets with ignore_idx)"
    def __init__(self, func, name="metric_func", ignore_idx=None, one_hot_encode_target=True, argmax_pred=False, softmax_pred=True, flatten_target=True, sigmoid_pred=False,metric_component=None):
        super().__init__()
        self.func = func
        self.ignore_idx = ignore_idx
        self.one_hot_encode_target = one_hot_encode_target
        self.argmax_pred = argmax_pred
        self.softmax_pred = softmax_pred
        self.flatten_target = flatten_target
        self.sigmoid_pred = sigmoid_pred
        self.metric_component = metric_component
        self.name=name

    def on_epoch_begin(self, **kwargs):
        self.y_pred = None
        self.y_true = None
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        #flatten everything (to make it also work for annotation tasks)
        y_pred_flat = last_output.view((-1,last_output.size()[-1]))
        
        if(self.flatten_target):
            y_true_flat = last_target.view(-1)
        y_true_flat = last_target

        #optionally take argmax of predictions
        if(self.argmax_pred is True):
            y_pred_flat = y_pred_flat.argmax(dim=1)
        elif(self.softmax_pred is True):
            y_pred_flat = F.softmax(y_pred_flat, dim=1)
        elif(self.sigmoid_pred is True):
            y_pred_flat = torch.sigmoid(y_pred_flat)
        
        #potentially remove ignore_idx entries
        if(self.ignore_idx is not None):
            selected_indices = (y_true_flat!=self.ignore_idx).nonzero().squeeze()
            y_pred_flat = y_pred_flat[selected_indices]
            y_true_flat = y_true_flat[selected_indices]
        
        y_pred_flat = to_np(y_pred_flat)
        y_true_flat = to_np(y_true_flat)

        if(self.one_hot_encode_target is True):
            y_true_flat = one_hot_np(y_true_flat,last_output.size()[-1])

        if(self.y_pred is None):
            self.y_pred = y_pred_flat
            self.y_true = y_true_flat
        else:
            self.y_pred = np.concatenate([self.y_pred, y_pred_flat], axis=0)
            self.y_true = np.concatenate([self.y_true, y_true_flat], axis=0)
    
    def on_epoch_end(self, last_metrics, **kwargs):
        #access full metric (possibly multiple components) via self.metric_complete
        self.metric_complete = self.func(self.y_true, self.y_pred)
        if(self.metric_component is not None):
            return add_metrics(last_metrics, self.metric_complete[self.metric_component])
        else:
            return add_metrics(last_metrics, self.metric_complete)

def fmax_metric(targs,preds):
    return evaluate_experiment(targs,preds)["Fmax"]

def auc_metric(targs,preds):
    return evaluate_experiment(targs,preds)["macro_auc"]

def mse_flat(preds,targs):
    return torch.mean(torch.pow(preds.view(-1)-targs.view(-1),2))

def nll_regression(preds,targs):
    #preds: bs, 2
    #targs: bs, 1
    preds_mean = preds[:,0]
    #warning: output goes through exponential map to ensure positivity
    preds_var = torch.clamp(torch.exp(preds[:,1]),1e-4,1e10)
    #print(to_np(preds_mean)[0],to_np(targs)[0,0],to_np(torch.sqrt(preds_var))[0])
    return torch.mean(torch.log(2*math.pi*preds_var)/2) + torch.mean(torch.pow(preds_mean-targs[:,0],2)/2/preds_var)
    
def nll_regression_init(m):
    assert(isinstance(m, nn.Linear))
    nn.init.normal_(m.weight,0.,0.001)
    nn.init.constant_(m.bias,4)

def lr_find_plot(learner, path, filename="lr_find", n_skip=10, n_skip_end=2):
    '''saves lr_find plot as file (normally only jupyter output)
    on the x-axis is lrs[-1]
    '''
    learner.lr_find()
    
    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("learning rate (log scale)")
    losses = [ to_np(x) for x in learner.recorder.losses[n_skip:-(n_skip_end+1)]]
    #print(learner.recorder.val_losses)
    #val_losses = [ to_np(x) for x in learner.recorder.val_losses[n_skip:-(n_skip_end+1)]]

    plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)],losses )
    #plt.plot(learner.recorder.lrs[n_skip:-(n_skip_end+1)],val_losses )

    plt.xscale('log')
    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)

def losses_plot(learner, path, filename="losses", last:int=None):
    '''saves lr_find plot as file (normally only jupyter output)
    on the x-axis is lrs[-1]
    '''
    backend_old= matplotlib.get_backend()
    plt.switch_backend('agg')
    plt.ylabel("loss")
    plt.xlabel("Batches processed")

    last = ifnone(last,len(learner.recorder.nb_batches))
    l_b = np.sum(learner.recorder.nb_batches[-last:])
    iterations = range_of(learner.recorder.losses)[-l_b:]
    plt.plot(iterations, learner.recorder.losses[-l_b:], label='Train')
    val_iter = learner.recorder.nb_batches[-last:]
    val_iter = np.cumsum(val_iter)+np.sum(learner.recorder.nb_batches[:-last])
    plt.plot(val_iter, learner.recorder.val_losses[-last:], label='Validation')
    plt.legend()

    plt.savefig(str(path/(filename+'.png')))
    plt.switch_backend(backend_old)

class fastai_model(ClassificationModel):
    def __init__(self,name,n_classes,freq,outputfolder,input_shape,pretrained=False,input_size=2.5,input_channels=12,chunkify_train=False,chunkify_valid=True,bs=128,ps_head=0.5,lin_ftrs_head=[128],wd=1e-2,epochs=50,lr=1e-2,kernel_size=5,loss="binary_cross_entropy",pretrainedfolder=None,n_classes_pretrained=None,gradual_unfreezing=True,discriminative_lrs=True,epochs_finetuning=30,early_stopping=None,aggregate_fn="max",concat_train_val=False):
        super().__init__()
        
        self.name = name
        self.num_classes = n_classes if loss!= "nll_regression" else 2
        self.target_fs = freq
        self.outputfolder = Path(outputfolder)

        self.input_size=int(input_size*self.target_fs)
        self.input_channels=input_channels

        self.chunkify_train=chunkify_train
        self.chunkify_valid=chunkify_valid

        self.chunk_length_train=2*self.input_size#target_fs*6
        self.chunk_length_valid=self.input_size

        self.min_chunk_length=self.input_size#chunk_length

        self.stride_length_train=self.input_size#chunk_length_train//8
        self.stride_length_valid=self.input_size//2#chunk_length_valid

        self.copies_valid = 0 #>0 should only be used with chunkify_valid=False
        
        self.bs=bs
        self.ps_head=ps_head
        self.lin_ftrs_head=lin_ftrs_head
        self.wd=wd
        self.epochs=epochs
        self.lr=lr
        self.kernel_size = kernel_size
        self.loss = loss
        self.input_shape = input_shape

        if pretrained == True:
            if(pretrainedfolder is None):
                pretrainedfolder = Path('../output/exp0/models/'+name.split("_pretrained")[0]+'/')
            if(n_classes_pretrained is None):
                n_classes_pretrained = 71
  
        self.pretrainedfolder = None if pretrainedfolder is None else Path(pretrainedfolder)
        self.n_classes_pretrained = n_classes_pretrained
        self.discriminative_lrs = discriminative_lrs
        self.gradual_unfreezing = gradual_unfreezing
        self.epochs_finetuning = epochs_finetuning

        self.early_stopping = early_stopping
        self.aggregate_fn = aggregate_fn
        self.concat_train_val = concat_train_val

    def fit(self, X_train, y_train, X_val, y_val):
        #convert everything to float32
        X_train = [l.astype(np.float32) for l in X_train]
        X_val = [l.astype(np.float32) for l in X_val]
        y_train = [l.astype(np.float32) for l in y_train]
        y_val = [l.astype(np.float32) for l in y_val]

        if(self.concat_train_val):
            X_train += X_val
            y_train += y_val
        
        if(self.pretrainedfolder is None): #from scratch
            print("Training from scratch...")
            learn = self._get_learner(X_train,y_train,X_val,y_val)
            
            #if(self.discriminative_lrs):
            #    layer_groups=learn.model.get_layer_groups()
            #    learn.split(layer_groups)
            learn.model.apply(weight_init)
            
            #initialization for regression output
            if(self.loss=="nll_regression" or self.loss=="mse"):
                output_layer_new = learn.model.get_output_layer()
                output_layer_new.apply(nll_regression_init)
                learn.model.set_output_layer(output_layer_new)
            
            lr_find_plot(learn, self.outputfolder)    
            learn.fit_one_cycle(self.epochs,self.lr)#slice(self.lr) if self.discriminative_lrs else self.lr)
            losses_plot(learn, self.outputfolder)
        else: #finetuning
            print("Finetuning...")
            #create learner
            learn = self._get_learner(X_train,y_train,X_val,y_val,self.n_classes_pretrained)
            
            #load pretrained model
            learn.path = self.pretrainedfolder
            learn.load(self.pretrainedfolder.stem)
            learn.path = self.outputfolder

            #exchange top layer
            output_layer = learn.model.get_output_layer()
            output_layer_new = nn.Linear(output_layer.in_features,self.num_classes).cuda()
            apply_init(output_layer_new, nn.init.kaiming_normal_)
            learn.model.set_output_layer(output_layer_new)
            
            #layer groups
            if(self.discriminative_lrs):
                layer_groups=learn.model.get_layer_groups()
                learn.split(layer_groups)

            learn.train_bn = True #make sure if bn mode is train
            
            
            #train
            lr = self.lr
            if(self.gradual_unfreezing):
                assert(self.discriminative_lrs is True)
                learn.freeze()
                lr_find_plot(learn, self.outputfolder,"lr_find0")
                learn.fit_one_cycle(self.epochs_finetuning,lr)
                losses_plot(learn, self.outputfolder,"losses0")
                #for n in [0]:#range(len(layer_groups)):
                #    learn.freeze_to(-n-1)
                #    lr_find_plot(learn, self.outputfolder,"lr_find"+str(n))
                #    learn.fit_one_cycle(self.epochs_gradual_unfreezing,slice(lr))
                #    losses_plot(learn, self.outputfolder,"losses"+str(n))
                    #if(n==0):#reduce lr after first step
                    #    lr/=10.
                    #if(n>0 and (self.name.startswith("fastai_lstm") or self.name.startswith("fastai_gru"))):#reduce lr further for RNNs
                    #    lr/=10
                    
            learn.unfreeze()
            lr_find_plot(learn, self.outputfolder,"lr_find"+str(len(layer_groups)))
            learn.fit_one_cycle(self.epochs_finetuning,slice(lr/1000,lr/10))
            losses_plot(learn, self.outputfolder,"losses"+str(len(layer_groups)))

        learn.save(self.name) #even for early stopping the best model will have been loaded again
    
    def predict(self, X):
        X = [l.astype(np.float32) for l in X]
        y_dummy = [np.ones(self.num_classes,dtype=np.float32) for _ in range(len(X))]
        
        learn = self._get_learner(X,y_dummy,X,y_dummy)
        learn.load(self.name)
        
        preds,targs=learn.get_preds()
        preds=to_np(preds)
        
        idmap=learn.data.valid_ds.get_id_mapping()

        return aggregate_predictions(preds,idmap=idmap,aggregate_fn = np.mean if self.aggregate_fn=="mean" else np.amax)  
        
    def _get_learner(self, X_train,y_train,X_val,y_val,num_classes=None):
        df_train = pd.DataFrame({"data":range(len(X_train)),"label":y_train})
        df_valid = pd.DataFrame({"data":range(len(X_val)),"label":y_val})
        
        tfms_ptb_xl = [ToTensor()]
                
        ds_train=TimeseriesDatasetCrops(df_train,self.input_size,num_classes=self.num_classes,chunk_length=self.chunk_length_train if self.chunkify_train else 0,min_chunk_length=self.min_chunk_length,stride=self.stride_length_train,transforms=tfms_ptb_xl,annotation=False,col_lbl ="label",npy_data=X_train)
        ds_valid=TimeseriesDatasetCrops(df_valid,self.input_size,num_classes=self.num_classes,chunk_length=self.chunk_length_valid if self.chunkify_valid else 0,min_chunk_length=self.min_chunk_length,stride=self.stride_length_valid,transforms=tfms_ptb_xl,annotation=False,col_lbl ="label",npy_data=X_val)
    
        db = DataBunch.create(ds_train,ds_valid,bs=self.bs)

        if(self.loss == "binary_cross_entropy"):
            loss = F.binary_cross_entropy_with_logits
        elif(self.loss == "cross_entropy"):
            loss = F.cross_entropy
        elif(self.loss == "mse"):
            loss = mse_flat
        elif(self.loss == "nll_regression"):
            loss = nll_regression    
        else:
            print("loss not found")
            assert(True)   
               
        self.input_channels = self.input_shape[-1]
        metrics = []

        print("model:",self.name) #note: all models of a particular kind share the same prefix but potentially a different postfix such as _input256
        num_classes = self.num_classes if num_classes is None else num_classes
        #resnet resnet1d18,resnet1d34,resnet1d50,resnet1d101,resnet1d152,resnet1d_wang,resnet1d,wrn1d_22
        if(self.name.startswith("fastai_resnet1d18")):
            model = resnet1d18(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d34")):
            model = resnet1d34(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d50")):
            model = resnet1d50(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d101")):
            model = resnet1d101(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d152")):
            model = resnet1d152(num_classes=num_classes,input_channels=self.input_channels,inplanes=128,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_resnet1d_wang")):
            model = resnet1d_wang(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_wrn1d_22")):    
            model = wrn1d_22(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        
        #xresnet ... (order important for string capture)
        elif(self.name.startswith("fastai_xresnet1d18_deeper")):
            model = xresnet1d18_deeper(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d34_deeper")):
            model = xresnet1d34_deeper(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d50_deeper")):
            model = xresnet1d50_deeper(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d18_deep")):
            model = xresnet1d18_deep(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d34_deep")):
            model = xresnet1d34_deep(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d50_deep")):
            model = xresnet1d50_deep(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d18")):
            model = xresnet1d18(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d34")):
            model = xresnet1d34(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d50")):
            model = xresnet1d50(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d101")):
            model = xresnet1d101(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_xresnet1d152")):
            model = xresnet1d152(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
                        
        #inception
        #passing the default kernel size of 5 leads to a max kernel size of 40-1 in the inception model as proposed in the original paper
        elif(self.name == "fastai_inception1d_no_residual"):#note: order important for string capture
            model = inception1d(num_classes=num_classes,input_channels=self.input_channels,use_residual=False,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head,kernel_size=8*self.kernel_size)
        elif(self.name.startswith("fastai_inception1d")):
            model = inception1d(num_classes=num_classes,input_channels=self.input_channels,use_residual=True,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head,kernel_size=8*self.kernel_size)


        #basic_conv1d fcn,fcn_wang,schirrmeister,sen,basic1d
        elif(self.name.startswith("fastai_fcn_wang")):#note: order important for string capture
            model = fcn_wang(num_classes=num_classes,input_channels=self.input_channels,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_fcn")):
            model = fcn(num_classes=num_classes,input_channels=self.input_channels)
        elif(self.name.startswith("fastai_schirrmeister")):
            model = schirrmeister(num_classes=num_classes,input_channels=self.input_channels,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_sen")):
            model = sen(num_classes=num_classes,input_channels=self.input_channels,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_basic1d")):    
            model = basic1d(num_classes=num_classes,input_channels=self.input_channels,kernel_size=self.kernel_size,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        #RNN
        elif(self.name.startswith("fastai_lstm_bidir")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=True,bidirectional=True,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_gru_bidir")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=False,bidirectional=True,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_lstm")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=True,bidirectional=False,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        elif(self.name.startswith("fastai_gru")):
            model = RNN1d(input_channels=self.input_channels,num_classes=num_classes,lstm=False,bidirectional=False,ps_head=self.ps_head,lin_ftrs_head=self.lin_ftrs_head)
        else:
            print("Model not found.")
            assert(True)
            
        learn = Learner(db,model, loss_func=loss, metrics=metrics,wd=self.wd,path=self.outputfolder)
        
        if(self.name.startswith("fastai_lstm") or self.name.startswith("fastai_gru")):
            learn.callback_fns.append(partial(GradientClipping, clip=0.25))

        if(self.early_stopping is not None):
            #supported options: valid_loss, macro_auc, fmax
            if(self.early_stopping == "macro_auc" and self.loss != "mse" and self.loss !="nll_regression"):
                metric = metric_func(auc_metric, self.early_stopping, one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True, flatten_target=False)
                learn.metrics.append(metric)
                learn.callback_fns.append(partial(SaveModelCallback, monitor=self.early_stopping, every='improvement', name=self.name))
            elif(self.early_stopping == "fmax" and self.loss != "mse" and self.loss !="nll_regression"):
                metric = metric_func(fmax_metric, self.early_stopping, one_hot_encode_target=False, argmax_pred=False, softmax_pred=False, sigmoid_pred=True, flatten_target=False)
                learn.metrics.append(metric)
                learn.callback_fns.append(partial(SaveModelCallback, monitor=self.early_stopping, every='improvement', name=self.name))
            elif(self.early_stopping == "valid_loss"):
                learn.callback_fns.append(partial(SaveModelCallback, monitor=self.early_stopping, every='improvement', name=self.name))
            
        return learn
