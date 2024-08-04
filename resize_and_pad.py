def resize_and_pad(img, input_size=MODEL_INPUT_SIZE):
    input_h, input_w = input_size
    ori_h, ori_w = img.shape[:2]
    ratio = min(input_h / ori_h, input_w / ori_w)
    
    img = F.interpolate(
        img.view(1, 1, ori_h, ori_w),
        mode="bilinear",
        scale_factor=ratio,
        recompute_scale_factor=True
        )[0, 0]
    
    padded_img = torch.zeros(
        (input_h, input_w),
        dtype=img.dtype,
        device='cuda'
        )
    
    cur_h, cur_w = img.shape
    y_start = (input_h - cur_h) // 2
    x_start = (input_w - cur_w) // 2
    padded_img[y_start:y_start + cur_h, x_start:x_start + cur_w] = img
    padded_img = padded_img.unsqueeze(-1).expand(-1, -1, 3)
    return padded_img