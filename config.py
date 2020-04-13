config = {}

config['data_path'] = '/your/path/train'
config['sequence_length'] = 10 * 100 # 3-10 * 1e2
config['dataset_feature'] = f'/your/path/LID-feature-{config["sequence_length"]}.hdf5'
config['dataset'] = f'/your/path/LID-dataset-{config["sequence_length"]}.hdf5'
config['resume'] = True#False
config['checkpoint'] = 'lid_model_0.90.hdf5'

config['debug'] = False
config['debug_size'] = 500

config['l2_regularizer'] = 1e-2
config['dropout'] = 0.5
config['learning_rate'] = 1e-3
config['batch_size'] = 256
config['epochs'] = 20
