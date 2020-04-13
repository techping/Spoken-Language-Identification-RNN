config = {}

config['data_path'] = '/home/tf-boys/docker_shared/train'
config['sequence_length'] = 10 * 100 # 3-10 * 1e2
config['dataset_feature'] = f'/home/tf-boys/docker_shared/LID-feature-{config["sequence_length"]}.hdf5'
config['dataset_original_feature'] = f'/home/tf-boys/docker_shared/LID-original-feature-{config["sequence_length"]}.hdf5'
config['dataset_original'] = f'/home/tf-boys/docker_shared/LID-original-dataset-{config["sequence_length"]}.hdf5'
config['dataset'] = f'/home/tf-boys/docker_shared/LID-dataset-{config["sequence_length"]}.hdf5'
config['dataset_balanced'] = f'/home/tf-boys/docker_shared/LID-dataset-balanced-{config["sequence_length"]}.hdf5'
config['dataset_original_balanced'] = f'/home/tf-boys/docker_shared/LID-dataset-original-balanced-{config["sequence_length"]}.hdf5'
config['resume'] = True#False
config['checkpoint'] = 'val-lstm-r-0.90.hdf5'#'best.hdf5'#'val-lstm-0.89.hdf5'#'best.hdf5'

config['debug'] = False
config['debug_size'] = 500

config['l2_regularizer'] = 1e-2
config['dropout'] = 0.5
config['learning_rate'] = 1e-3
config['batch_size'] = 256
config['epochs'] = 20
