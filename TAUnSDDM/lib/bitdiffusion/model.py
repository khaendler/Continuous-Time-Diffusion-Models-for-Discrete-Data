from bit_diffusion.bit_diffusion import *


class BitDiffusionSubset(BitDiffusion):
    """ A bit diffusion that works together with networks from this repo. Does not have self conditioning."""
    def __init__(
            self,
            model,  # any nn.Module taking (x, t)
            image_size,
            timesteps=1000,
            use_ddim=False,
            noise_schedule='cosine',
            time_difference=0.,
            bit_scale=1.,
            bits=8
    ):
        super().__init__(
            model=model,
            image_size=image_size,
            timesteps=timesteps,
            use_ddim=use_ddim,
            noise_schedule=noise_schedule,
            time_difference=time_difference,
            bit_scale=bit_scale
        )
        self.bits=bits

    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.timesteps):
            # add the time delay
            time_next = (time_next - time_difference).clamp(min=0.)

            # get predicted x0
            x_start = self.model(img, time)

            # clip x0
            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            # get alpha sigma of time and next time
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance
            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return bits_to_decimal(img, bits=self.bits)

    @torch.no_grad()
    def ddim_sample(self, shape, time_difference=None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)

        time_pairs = self.get_sampling_timesteps(batch, device=device)

        img = torch.randn(shape, device=device)

        x_start = None

        for times, times_next in tqdm(time_pairs, desc='sampling loop time step'):
            # add the time delay

            times_next = (times_next - time_difference).clamp(min=0.)

            # get times and noise levels
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # predict x0
            x_start = self.model(img, times)

            # clip x0
            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get predicted noise
            pred_noise = (img - alpha * x_start) / sigma.clamp(min=1e-8)

            # calculate x next
            img = x_start * alpha_next + pred_noise * sigma_next

        return bits_to_decimal(img, bits=self.bits)

    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # sample random times
        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.)

        # convert image to bit representation
        img = decimal_to_bits(img, bits=self.bits) * self.bit_scale

        # noise sample
        noise = torch.randn_like(img)

        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)

        noised_img = alpha * img + sigma * noise

        # predict and take gradient step
        pred = self.model(noised_img, times)

        return F.mse_loss(pred, img)