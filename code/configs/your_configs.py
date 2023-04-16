conf_tf_inception = {'modelname':'tf_inception', 'modeltype':'inception_time_model', 
    'parameters':dict()}


conf_tf_inception_all = {'modelname':'tf_inception_all', 'modeltype':'inception_time_model', 
    'parameters':dict(epoch=15, batch_size=16, lr_init=0.001, lr_red="no", model_depth=9 , loss="bce" , kernel_size=60)}

conf_tf_inception_diagnostic = {'modelname':'tf_inception_diagnostic', 'modeltype':'inception_time_model', 
    'parameters':dict(epoch=25, batch_size=32, lr_init=0.001 ,lr_red="no", model_depth=6 , loss="bce" , kernel_size=60)}

conf_tf_inception_form = {'modelname':'tf_inception_form', 'modeltype':'inception_time_model', 
    'parameters':dict(epoch=25, batch_size=64, lr_init=0.001 ,lr_red="no", model_depth=6 , loss="bce" , kernel_size=20)}

conf_tf_inception_rhythm = {'modelname':'tf_inception_rhythm', 'modeltype':'inception_time_model', 
    'parameters':dict(epoch=25, batch_size=16, lr_init=0.001, lr_red="no", model_depth=9 , loss="wbce" , kernel_size=40)}

conf_tf_inception_subdiagnostic = {'modelname':'tf_inception_subdiagnostic', 'modeltype':'inception_time_model', 
    'parameters':dict(epoch=15, batch_size=64, lr_init=0.001 , lr_red="no", model_depth=6 , loss="wbce" , kernel_size=20)}

conf_tf_inception_superdiagnostic = {'modelname':'tf_inception_superdiagnostic', 'modeltype':'inception_time_model', 
    'parameters':dict(epoch=25, batch_size=64, lr_init=0.001 ,lr_red="yes", model_depth=12 , loss="bce" , kernel_size=40)}

