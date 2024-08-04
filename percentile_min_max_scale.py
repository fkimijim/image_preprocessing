def percentile_min_max_scale(img, pct=99):
    if isinstance(img, np.ndarray):
        maxv = np.percentile(img, pct) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv
        ret = np.clip(ret, 0, 1)
    elif isinstance(img, torch.Tensor):
        maxv = torch.quantile(img, pct / 100) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv
        ret = torch.clamp(ret, 0, 1)
    else:
        raise ValueError('Invalid img type, should be numpy array or torch.Tensor')
    return ret