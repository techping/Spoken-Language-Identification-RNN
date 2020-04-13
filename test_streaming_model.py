from config import config
from model import get_model, get_streaming_model, my_sparse_categorical_accuracy
from data import get_data
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import tensorflow.keras.backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

_, _, X_val, y_val = get_data(valid=True)
print('data done')

training_model = get_model()
training_model.load_weights(config['checkpoint'])
streaming_model = get_streaming_model(config['checkpoint'])
print('model done')


num_seqs = y_val.shape[0]
train_seq_length = config['sequence_length']
feature_dim = 64

if False:
    ##### demo the behaivor
    print('\n\n******the streaming-inference model can replicate the sequence-based trained model:\n')
    for s in range(num_seqs):
        print(f'\n\nRunning Sequence {s} with STATE RESET:\n')
        in_seq = X_val[s].reshape((1, train_seq_length, feature_dim))
        seq_pred = training_model.predict(in_seq)
        seq_pred = tf.argmax(seq_pred, axis=2)[0]
        for n in tqdm(range(train_seq_length)):
            in_feature_vector = X_val[s][n].reshape((1, 1, feature_dim))
            single_pred = streaming_model.predict(in_feature_vector)
            single_pred = tf.argmax(single_pred, axis=2)[0]
            assert np.isclose(single_pred[0], seq_pred[n]), 'model error'
        streaming_model.reset_states()

    print('\n\n******streaming-inference state needs reset between sequences to replicate sequence-based trained model:\n')
    neq_count = 0
    for s in range(num_seqs):
        print(f'\n\nRunning Sequence {s} with NO STATE RESET:\n')
        in_seq = X_val[s].reshape((1, train_seq_length, feature_dim))
        seq_pred = training_model.predict(in_seq)
        seq_pred = tf.argmax(seq_pred, axis=2)[0]
        for n in tqdm(range(train_seq_length)):
            in_feature_vector = X_val[s][n].reshape((1, 1, feature_dim))
            single_pred = streaming_model.predict(in_feature_vector)
            single_pred = tf.argmax(single_pred, axis=2)[0]
            if not np.isclose(single_pred[0], seq_pred[n]):
                neq_count += 1
            #print(f'{s},{n}:{np.isclose(single_pred[0], seq_pred[n])}')
    print(f'neq_count: {neq_count}')

# input random seqence length
correct = 0
count = 0
ommited_silence_len = 0
res = [np.array([]) for _ in range(X_val.shape[0])]
for i in tqdm(range(X_val.shape[0])):
    for j in range(X_val.shape[1]):
        if int(y_val[i][j][0]) == -1:
            continue # omit silence signals
        count += 1
        X = X_val[i][j].reshape((1, 1, feature_dim))
        #y = y_val[i][j].reshape((1, 1, 1))
        #pred = streaming_model.predict(X, use_multiprocessing=True, verbose=)
        pred = streaming_model(X)
        if res[i].size == 0:
            res[i] = pred[0].numpy()
        else:
            res[i] = np.vstack((res[i], pred[0].numpy()))
        if tf.argmax(pred[0][0]).numpy() == int(y_val[i][j][0]):
            correct += 1
        #acc = streaming_model.evaluate(x=X, y=y, verbose=0, use_multiprocessing=True)[1]
        K.clear_session()
    #print(correct / count)
    #print(res[i].shape)
    #print(f'Sequence {i+1} acc: {sub_correct / sub_ommited_silence_len}')
    streaming_model.reset_states()
print(f'Total streaming accuracy on validation set: {correct / count}')

import matplotlib.pyplot as plt

# randomly sample 10 files to plot their output over time
N = np.random.randint(X_val.shape[0], size=10)
for i in range(10):
    plt.figure()
    plt.title(f'output over time-{N[i]}')
    plt.xlabel('time')
    plt.ylabel('probability')
    for j in range(3):
        plt.plot([k for k in range(res[N[i]].shape[0])], res[N[i]].T[i], label=f'prob-{j}')
    plt.legend()
    plt.savefig(f'output-{N[i]}.png')
