import argparse
import numpy as np
import torch
import multiprocessing as mp

import TAUnSDDM.lib.datasets.dataset_utils as dataset_utils
from TAUnSDDM.lib.bitdiffusion.trainer import Trainer
from TAUnSDDM.config.config_bitdiffusion import get_config
from TAUnSDDM.lib.datasets.maze import maze_acc, Maze3SForAnalogBits
from TAUnSDDM.lib.datasets.metrics import compute_hellinger


def main(config_name):
    cfg = get_config(config_name)

    net = cfg.model.net_class(cfg)
    model = cfg.model.model_class(
        model=net,
        image_size=cfg.data.image_size,
        timesteps=cfg.model.timesteps,
        use_ddim=cfg.model.use_ddim,
        noise_schedule=cfg.model.noise_schedule,
        time_difference=cfg.model.time_difference,
        bit_scale=cfg.model.bit_scale,
        bits=cfg.data.bits
    )
    print(model.device)
    cfg.data.limit = 1
    dataset = dataset_utils.get_dataset(cfg, cfg.device)
    trainer = Trainer(
        diffusion_model=model,
        dataset=dataset,
        train_batch_size=cfg.data.batch_size,
        gradient_accumulate_every=cfg.training.gradient_accumulate_every,
        train_lr=cfg.training.train_lr,
        train_num_steps=cfg.training.train_num_steps,
        ema_update_every=cfg.training.ema_update_every,
        ema_decay=cfg.training.ema_decay,
        adam_betas=cfg.training.adam_betas,
        save_and_sample_every=cfg.training.save_and_sample_every,
        num_samples=cfg.training.num_samples,
        results_folder=cfg.training.results_folder,
        amp=cfg.training.amp,
        mixed_precision_type=cfg.training.mixed_precision_type,
        split_batches=cfg.training.split_batches
    )

    trainer.load('29')
    model = trainer.ema.ema_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    gen_samples = model.sample(batch_size=5000)
    gen_samples = gen_samples.squeeze(1)
    gen_samples_int = (gen_samples * 2).round().cpu().numpy().astype(np.int32)
    maze_results = maze_acc(cfg, gen_samples_int)

    cfg.data.limit = 5000
    dataset = dataset_utils.get_dataset(cfg, device="cpu")
    real_samples = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    real_samples = real_samples.squeeze(1).numpy()
    h_dist = compute_hellinger(gen_samples_int, real_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BitDiffusion model.")
    parser.add_argument(
        "--config_name",
        type=str,
        default="SudokuBitDiffusion",
        help="Name of the configuration to load."
    )

    args = parser.parse_args()

    mp.set_start_method('spawn', force=True)
    main(args.config_name)
