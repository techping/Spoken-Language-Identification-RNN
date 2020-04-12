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
ommited_silence_len = 0
for i in tqdm(range(X_val.shape[0])):
    count = 0
    sub_correct = 0
    sub_ommited_silence_len = 0
    while count < train_seq_length:
        N = np.random.randint(300, 1000) # randomly generate sequence length
        if N + count > train_seq_length:
            N = train_seq_length - count
        X = X_val[i][count:count + N].reshape((1, -1, feature_dim))
        y = y_val[i][count:count + N].reshape((1, -1, 1))
        #pred = streaming_model.predict(X, use_multiprocessing=True, verbose=)
        pred = streaming_model(X)
        cor = tf.reduce_sum(my_sparse_categorical_accuracy(y, pred))
        sub_ommited_silence_len += np.sum(tf.not_equal(tf.cast(y, tf.int32), -1 * tf.ones((1,), dtype=tf.int32)).numpy().reshape((-1)))
        #print(f'acc: {cor / N}')
        #acc = streaming_model.evaluate(x=X, y=y, verbose=0, use_multiprocessing=True)[1]
        sub_correct += cor#int(acc * N)
        count += N
        K.clear_session()
    correct += sub_correct
    ommited_silence_len += sub_ommited_silence_len
    #print(f'Sequence {i+1} acc: {sub_correct / sub_ommited_silence_len}')
    streaming_model.reset_states()
print(f'Total streaming accuracy on validation set: {correct / ommited_silence_len}')
