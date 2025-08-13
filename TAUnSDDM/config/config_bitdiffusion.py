import os
import ml_collections

import TAUnSDDM.lib.datasets.maze
from TAUnSDDM.lib.networks.ddsm_networks import ProteinScoreNet
from TAUnSDDM.lib.bitdiffusion.model import BitDiffusionSubset


def get_config(name):
    config = ml_collections.ConfigDict()
    if name == 'MazeBitDiffusion':
        save_directory = "./results"

        config.device = "cpu"
        config.distributed = False
        config.num_gpus = 0

        config.training = training = ml_collections.ConfigDict()
        training.gradient_accumulate_every = 1
        training.train_lr = 1.5e-4
        training.adam_betas = (0.9, 0.999)
        training.train_num_steps = 300000  # 0  # 2000 #2000000
        training.ema_update_every = 20
        training.ema_decay = 0.9999
        training.save_and_sample_every = 10000
        training.num_samples = 25
        training.results_folder = save_directory
        training.amp = False
        training.mixed_precision_type = 'fp16'
        training.split_batches = True
        training.resume = True

        config.data = data = ml_collections.ConfigDict()
        data.name = 'Maze3SForAnalogBits'
        data.batch_size = 128
        data.bits = 8
        data.S = data.bits
        data.image_size = 15
        data.shape = [1, data.image_size, data.image_size]
        data.crop_wall = False
        data.limit = 1
        data.random_transform = True

        config.model = model = ml_collections.ConfigDict()
        model.concat_dim = data.shape[0]
        model.model_class = BitDiffusionSubset
        model.timesteps = 1000
        model.use_ddim = False
        model.noise_schedule = 'cosine'
        model.time_difference = 0.
        model.bit_scale = 1.

        model.net_class = ProteinScoreNet
        model.embed_dim = 200

        config.network = network = ml_collections.ConfigDict()


    return config
