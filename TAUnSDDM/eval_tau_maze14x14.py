import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml.scalarfloat import ScalarFloat

import lib.utils.bookkeeping as bookkeeping
import lib.datasets.dataset_utils as dataset_utils
import TAUnSDDM.lib.sampling.sampling
import lib.sampling.sampling_utils as sampling_utils

from TAUnSDDM.lib.models.models import UniVarProteinScoreNetEMA
from lib.datasets.maze import Maze3SComplete
from lib.utils.bookkeeping import load_config
from lib.utils.bookkeeping import load_state

from TAUnSDDM.lib.datasets.metrics import compute_hellinger


def main():
    # Paths and configuration
    model_date = "2025-09-15"
    model_name = "model_299999"
    save_location = os.path.join("TAUnSDDM", "SavedModels", "MAZEelbo", model_date)
    checkpoint_path = os.path.join(save_location, model_name)
    config_name = "config_001.yaml"
    cfg_location = os.path.join("TAUnSDDM", "SavedModels", "MAZEelbo", "2025-09-14")
    config_path = os.path.join(cfg_location, config_name)

    # Load configuration
    cfg = bookkeeping.load_config(config_path)
    cfg.device = "cpu"  # force CPU
    cfg.data.limit = 5000  # limit dataset to 5000 samples

    device = torch.device(cfg.device)

    # Load model
    model = UniVarProteinScoreNetEMA(cfg, device, None)
    optimizer = torch.optim.Adam(model.parameters(), cfg.optimizer.lr)
    state = {"model": model, "optimizer": optimizer, "n_iter": 0}

    state = bookkeeping.load_state(state, checkpoint_path, device)

    # Dataset
    dataset = Maze3SComplete(cfg, device="cpu", _=None)
    real_samples = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    real_samples = real_samples.squeeze(1).numpy()

    # Sampler
    sampler = sampling_utils.get_sampler(cfg)
    n_samples = 5000

    # Generate samples
    model.eval()
    gen_samples, _ = sampler.sample(state["model"], n_samples)
    gen_samples_int = gen_samples.cpu().numpy().astype(np.float32)  # convert to numpy for metric

    # Hellinger distance
    h_dist = compute_hellinger(gen_samples_int, real_samples)
    print(f"Hellinger distance between generated and real samples: {h_dist}")


if __name__ == "__main__":
    main()
