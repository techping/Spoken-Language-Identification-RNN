from config import config
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, TimeDistributed
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

# LR_START = 0.00001
# LR_MAX = 0.00005 #* #strategy.num_replicas_in_sync
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 5
# LR_SUSTAIN_EPOCHS = 3
# LR_EXP_DECAY = 0.5
# 
# def lrfn(epoch):
#     if epoch < LR_RAMPUP_EPOCHS:
#         lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
#     elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
#         lr = LR_MAX
#     else:
#         lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
#     return lr
#                                                         
# lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

def my_sparse_categorical_crossentropy(y_true, y_pred):
    #print(y_true.shape)
    #print(y_pred.shape)
    #weights = tf.convert_to_tensor([0.51206135, 1.7253448,  0.72872305])
    #print(weights)
    # omit silence
    mask = tf.not_equal(tf.cast(y_true, tf.int32), -1 * tf.ones((1,), dtype=tf.int32))
    #print(mask.shape)
    #mask = tf.reshape(mask, (-1, mask.shape[1],))
    mask = tf.squeeze(mask, axis=-1)
    omitted_y_true = tf.boolean_mask(y_true, mask)
    omitted_y_pred = tf.boolean_mask(y_pred, mask)
    loss = tf.keras.backend.sparse_categorical_crossentropy(omitted_y_true, omitted_y_pred)
    return loss
    #return loss * weights / K.sum(weights)

def my_sparse_categorical_accuracy(y_true, y_pred):
    #print('1')
    #print(weights)
    #print(y_true.shape)
    #print(y_pred.shape)
    # omit silence
    mask = tf.not_equal(tf.cast(y_true, tf.int32), -1 * tf.ones((1,), dtype=tf.int32))
    #print(mask.shape)
    #mask = tf.reshape(mask, (-1, mask.shape[1],))
    mask = tf.squeeze(mask, axis=-1)
    omitted_y_true = tf.boolean_mask(y_true, mask)
    omitted_y_pred = tf.boolean_mask(y_pred, mask)
    return tf.keras.metrics.sparse_categorical_accuracy(omitted_y_true, omitted_y_pred)

def get_model():
    model_in = Input(shape=(config['sequence_length'], 64))
    x = GRU(64, return_sequences=True, stateful=False)(model_in)
    x = GRU(32, return_sequences=True, stateful=False)(x)
    x = Dense(100, activation='tanh')(x)
    pred = Dense(3, activation='softmax')(x)
    model = Model(inputs=model_in, outputs=pred)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'], optimizer=tf.keras.optimizers.Adam(lr=config['learning_rate'], decay=1e-4))
    model.summary()
    plot_model(model, to_file=f'model-{config["sequence_length"]}.png', show_shapes=True, show_layer_names=True)
    return model
