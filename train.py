from config import config
from model import get_model
from data import get_data
import tensorflow as tf
from sklearn.utils import class_weight
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

X_train, y_train, X_val, y_val = get_data(size=None if config['debug'] is not True else config['debug_size'], batch_size=config['batch_size'])
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train[y_train!=-1].reshape((-1,))), y_train[y_train!=-1].reshape((-1,)))
#class_weights = [0.51206135, 1.7253448, 0.72872305]
print(class_weights)
print('data done')

model = get_model()
mcp = tf.keras.callbacks.ModelCheckpoint(monitor='val_my_sparse_categorical_accuracy', filepath='best.hdf5', verbose=1, save_best_only=True)
if config['resume'] == True:
    model.load_weights(config['checkpoint'])
print('model done')

try:
    model.fit(#x=X_val, y=y_val,
            x=X_train, y=y_train, 
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            batch_size=config['batch_size'], epochs=config['epochs'], callbacks=[mcp], use_multiprocessing=True, shuffle=True)
finally:
    pass
    #model.evaluate(x=X_val, y=y_val, batch_size=config['batch_size'], use_multiprocessing=True)
