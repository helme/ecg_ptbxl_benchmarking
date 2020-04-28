conf_wavelet_standard_lr = {'modelname':'Wavelet+LR', 'modeltype':'WAVELET', 
    'parameters':dict(
        regularizer_C=.001,
        classifier='LR'
    )}

conf_wavelet_standard_rf = {'modelname':'Wavelet+RF', 'modeltype':'WAVELET', 
    'parameters':dict(
        regularizer_C=.001,
        classifier='RF'
    )}


conf_wavelet_standard_nn = {'modelname':'Wavelet+NN', 'modeltype':'WAVELET', 
    'parameters':dict(
        regularizer_C=.001,
        classifier='NN'
    )}