from config import config
import os
import librosa
import numpy as np
from tqdm import tqdm
import h5py

def generate_features(valid=0.2):
    if os.path.isfile(config['dataset']):
        print('dataset already exist')
        return
    print(config['dataset_feature'])
    features = [np.array([]) for _ in range(3)]
    labels = [np.array([]) for _ in range(3)]
    save_fcn = lambda x, f : f if x.size == 0 else np.vstack((x, f))
    langs = ['train_english', 'train_hindi', 'train_mandarin']
    if not os.path.isfile(config['dataset_feature']):
        for i, lang in enumerate(langs):
            print(f'start process {lang[6:]}')
            files = os.listdir(os.path.join(config['data_path'], lang))
            for f in tqdm(sorted(files)):
                y, sr = librosa.load(os.path.join(config['data_path'], lang, f), sr=16000)
                intvls = librosa.effects.split(y)
                feat = np.array([])
                label = np.array([])
                for j, intvl in enumerate(intvls):
                    #label = i
                    if j == 0 and intvl[0] != 0:
                        feature = librosa.feature.mfcc(y=y[:intvl[0]], sr=sr, n_mfcc=64, n_fft=int(sr * 0.025), hop_length=int(sr * 0.010))
                        #label = -1 # mark silence
                        feat = save_fcn(feat, feature.T)
                        label = save_fcn(label, (-1 * np.ones((feature.shape[1], 1))))
                    feature = librosa.feature.mfcc(y=y[intvl[0]:intvl[1]], sr=sr, n_mfcc=64, n_fft=int(sr * 0.025), hop_length=int(sr * 0.010))
                    feat = save_fcn(feat, feature.T)
                    label = save_fcn(label, (i * np.ones((feature.shape[1], 1))))
                    if intvl[1] != len(y):
                        feature = librosa.feature.mfcc(y=y[intvl[1]:intvls[j + 1][0] if j + 1 < intvls.shape[0] else len(y)], sr=sr, n_mfcc=64, n_fft=int(sr * 0.025), hop_length=int(sr * 0.010))
                        #label = -1 # mark silence
                        feat = save_fcn(feat, feature.T)
                        label = save_fcn(label, (-1 * np.ones((feature.shape[1], 1))))
                rows = (feat.shape[0] // config['sequence_length']) * config['sequence_length']
                features[i] = save_fcn(features[i], feat[:rows].reshape((-1, config['sequence_length'], 64)))
                labels[i] = save_fcn(labels[i], label[:rows].reshape((-1, config['sequence_length'], 1)))
            print(features[i].shape)
        with h5py.File(config['dataset_feature'], 'w') as hf:
            for i in range(3):
                hf.create_dataset(f'features[{i}]', data=features[i])
                hf.create_dataset(f'labels[{i}]', data=labels[i])
    else:
        print('loading features...')
        # make random reproducible
        np.random.seed(976)
        with h5py.File(config['dataset_feature'], 'r') as hf:
            for i in range(3):
                print(f'features[{i}]...')
                features[i] = hf[f'features[{i}]'][:]
                labels[i] = hf[f'labels[{i}]'][:]
                perm = np.random.permutation(features[i].shape[0])
                features[i] = features[i][perm]
                labels[i] = labels[i][perm]
    X_train = np.concatenate((features[0][:int((1 - valid) * features[0].shape[0])], features[1][:int((1 - valid) * features[1].shape[0])], features[2][:int((1 - valid) * features[2].shape[0])]), axis=0)
    y_train = np.concatenate((labels[0][:int((1 - valid) * labels[0].shape[0])], labels[1][:int((1 - valid) * labels[1].shape[0])], labels[2][:int((1 - valid) * labels[2].shape[0])]), axis=0)
    X_val = np.concatenate((features[0][int((1 - valid) * features[0].shape[0]):], features[1][int((1 - valid) * features[1].shape[0]):], features[2][int((1 - valid) * features[2].shape[0]):]), axis=0)
    y_val = np.concatenate((labels[0][int((1 - valid) * labels[0].shape[0]):], labels[1][int((1 - valid) * labels[1].shape[0]):], labels[2][int((1 - valid) * labels[2].shape[0]):]), axis=0)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    with h5py.File(config['dataset'], 'w') as hf:
        hf.create_dataset('X_train', data=X_train)
        hf.create_dataset('y_train', data=y_train)
        hf.create_dataset('X_val', data=X_val)
        hf.create_dataset('y_val', data=y_val)

if __name__ == '__main__':
    generate_features()
