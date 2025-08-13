import multiprocessing as mp

import TAUnSDDM.lib.datasets.dataset_utils as dataset_utils
from TAUnSDDM.lib.bitdiffusion.trainer import Trainer
from TAUnSDDM.config.config_bitdiffusion import get_config

def main():
    cfg = get_config('MazeBitDiffusion')

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

    trainer.train()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()