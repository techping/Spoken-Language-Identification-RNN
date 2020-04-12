from config import config
from model import get_model, get_streaming_model
from data import get_data
import tensorflow as tf
from tqdm import tqdm
import numpy as np

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

#count = 0
#X_val = X_val.reshape((-1, 64))
#y_val = y_val.reshape((-1, 1))
correct = 0
for i in range(X_val.shape[0]):
    count = 0
    while count < train_seq_length:
        N = np.random.randint(300, 1000)
        if N + count > train_seq_length:
            N = train_seq_length - count
        X = X_val[i][count:count + N].reshape((1, -1, feature_dim))
        y = y_val[i][count:count + N].reshape((1, -1, 1))
        acc = streaming_model.evaluate(x=X, y=y, use_multiprocessing=True)[1]
        correct += (acc * N)
        count += N
    streaming_model.reset_states()
# while count < num_seqs * train_seq_length:
#     N = np.random.randint(300, 1000)# randomly generate sequence length
#     if N + count > num_seqs * train_seq_length: 
#         N = num_seqs * train_seq_length  - count
#     X = X_val[count:count+N].reshape((1, -1, feature_dim))
#     y = y_val[count:count+N].reshape((1, -1, 1))
#     acc = streaming_model.evaluate(x=X, y=y, use_multiprocessing=True)
#     #print(f'\racc: {acc}', end='')
#     correct += acc[1] * N
#     count += N
print(f'total streaming accuracy on validation set: {correct / (num_seqs * train_seq_length)}')
