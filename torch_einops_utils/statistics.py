from torch_einops_utils.torch_einops_utils import exists, masked_mean

def z_score(
    t,
    mask = None,
    dim = None,
    eps = 1e-5
):
    mean = masked_mean(t, mask = mask, dim = dim, keepdim = True)
    var = masked_mean((t - mean) ** 2, mask = mask, dim = dim, keepdim = True)
    out = (t - mean) * (var + eps).rsqrt()

    if not exists(mask):
        return out

    return out.masked_fill(~mask, 0.)
