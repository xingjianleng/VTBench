def forward(
    self,
    sample,
    sample_posterior=False,
    return_dict=True,
    generator=None,
):
    r"""
    Args:
        sample (`torch.Tensor`): Input sample.
        sample_posterior (`bool`, *optional*, defaults to `False`):
            Whether to sample from the posterior.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
    """
    x = sample
    posterior = self.encode(x).latent_dist
    if sample_posterior:
        z = posterior.sample(generator=generator)
    else:
        z = posterior.mode()
    dec = self.decode(z).sample
    return dec, None, None
