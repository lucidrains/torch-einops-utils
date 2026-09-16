from collections import namedtuple

from torch_einops_utils.torch_einops_utils import exists, pad_right_ndim_to, masked_mean

ZScoreStats = namedtuple('ZScoreStats', ['mean', 'var'])

def z_score(
    t,
    mask = None,
    dim = None,
    eps = 1e-5,
    return_stats = False,
    return_only_stats = False
):
    mean = masked_mean(t, mask = mask, dim = dim, keepdim = True)
    var = masked_mean((t - mean) ** 2, mask = mask, dim = dim, keepdim = True)

    stats = ZScoreStats(mean, var)

    if return_only_stats:
        return stats

    out = (t - mean) * (var + eps).rsqrt()

    if exists(mask):
        padded_mask = pad_right_ndim_to(mask, out.ndim)
        out = out.masked_fill(~padded_mask, 0.)

    if not return_stats:
        return out

    return out, stats
