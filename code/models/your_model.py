from models.base_model import ClassificationModel
import tensorflow as tf
import numpy as np


class inception_time_model(ClassificationModel):
    def __init__(self, name, n_classes,  sampling_frequency, outputfolder, input_shape):
        super(inception_time_model, self).__init__()
        self.name = name
        self.n_classes = n_classes
        self.sampling_frequency = sampling_frequency
        self.outputfolder = outputfolder
        self.input_shape = input_shape
        self.model = build_model((self.sampling_frequency*10,12),self.n_classes)
        self.epoch = 2 
        self.batch_size =32

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size, 
        validation_data=(X_val, y_val), callbacks = [tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)])
        #self.model.save(self.outputfolder +'last_model.h5')
    def predict(self, X):
        X = [l.astype(np.float32) for l in X]
        return self.model.predict(X)


def _inception_module(input_tensor, stride=1, activation='linear', use_bottleneck=True, kernel_size=40, bottleneck_size=64, nb_filters=64):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = tf.keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor

    # kernel_size_s = [3, 5, 8, 11, 17]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                              strides=stride, padding='same', activation=activation, use_bias=False)(
            input_inception))

    max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = tf.keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                  padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = tf.keras.layers.Concatenate(axis=2)(conv_list)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation='relu')(x)
    return x

def _shortcut_layer(input_tensor, out_tensor):
    shortcut_y = tf.keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                      padding='same', use_bias=False)(input_tensor)
    shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

    x = tf.keras.layers.Add()([shortcut_y, out_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_model(input_shape, nb_classes, depth=10, use_residual=True):
    input_layer = tf.keras.layers.Input(input_shape)

    x = input_layer
    input_res = input_layer

    for d in range(depth):

        x = _inception_module(x)

        if use_residual and d % 3 == 2:
            x = _shortcut_layer(input_res, x)
            input_res = x

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(x)

    output_layer = tf.keras.layers.Dense(units=nb_classes,activation='sigmoid')(gap_layer)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC(
                        num_thresholds=200,
                        curve='ROC',
                        summation_method='interpolation',
                        name="ROC",
                        multi_label=True,
                        ),
                       tf.keras.metrics.AUC(
                        num_thresholds=200,
                        curve='PR',
                        summation_method='interpolation',
                        name="PRC",
                        multi_label=True,
                        )
              ])
    print("Inception model built.")
    return model

def scheduler(epoch, lr):
    if epoch == 15:
        return lr*0.1
    elif epoch == 30:
        return lr*0.1
    elif epoch == 30:
        return lr*0.1
    else:
        return lr