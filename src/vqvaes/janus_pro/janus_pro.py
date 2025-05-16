def forward(self, input):
    quant, diff, [_, _, img_toks] = self.encode(input)

    batch_size, height, width, n_channel = (
        input.shape[0],
        quant.shape[-1],
        quant.shape[-2],
        quant.shape[-3],
    )
    codebook_entry = self.quantize.get_codebook_entry(
        img_toks, (batch_size, n_channel, height, width)
    )
    pixels = self.decode(codebook_entry)

    return pixels, img_toks, quant
