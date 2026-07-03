import numpy as np


def compute_mae(pred, gt):
    pred = pred.astype(np.float64) / 255.0 if pred.max() > 1 else pred.astype(np.float64)
    gt = gt.astype(np.float64) / 255.0 if gt.max() > 1 else gt.astype(np.float64)
    return float(np.mean(np.abs(pred - gt)))


def compute_smeasure(pred, gt, alpha=0.5):
    pred = pred.astype(np.float64) / 255.0 if pred.max() > 1 else pred.astype(np.float64)
    gt = (gt > 0.5).astype(np.float64) if gt.max() <= 1 else (gt > 127).astype(np.float64)

    gt_mean = gt.mean()
    if gt_mean == 0:
        s_obj = 1.0 - pred.mean()
    elif gt_mean == 1:
        s_obj = pred.mean()
    else:
        s_obj = _s_object(pred, gt)

    s_reg = _s_region(pred, gt)
    return alpha * s_obj + (1 - alpha) * s_reg


def _s_object(pred, gt):
    fg = pred[gt == 1]
    bg = pred[gt == 0]
    o_fg = 2 * fg.mean() / (fg.mean() ** 2 + 1.0 + 1e-8) if fg.size > 0 else 0.0
    o_bg = 2 * (1 - bg.mean()) / ((1 - bg.mean()) ** 2 + 1.0 + 1e-8) if bg.size > 0 else 0.0
    w = gt.sum() / gt.size
    return w * o_fg + (1 - w) * o_bg


def _s_region(pred, gt):
    x = int(np.round(gt.shape[1] * gt.mean(axis=0).mean()))
    y = int(np.round(gt.shape[0] * gt.mean(axis=1).mean()))
    x = np.clip(x, 1, gt.shape[1] - 1)
    y = np.clip(y, 1, gt.shape[0] - 1)

    w1 = x * y / gt.size
    w2 = (gt.shape[1] - x) * y / gt.size
    w3 = x * (gt.shape[0] - y) / gt.size
    w4 = 1.0 - w1 - w2 - w3

    q1 = _ssim(pred[:y, :x], gt[:y, :x])
    q2 = _ssim(pred[:y, x:], gt[:y, x:])
    q3 = _ssim(pred[y:, :x], gt[y:, :x])
    q4 = _ssim(pred[y:, x:], gt[y:, x:])

    return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4


def _ssim(pred, gt):
    if pred.size == 0:
        return 0.0
    mu_p, mu_g = pred.mean(), gt.mean()
    sigma_p = pred.std()
    sigma_g = gt.std()
    sigma_pg = ((pred - mu_p) * (gt - mu_g)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_p * mu_g + c1) * (2 * sigma_pg + c2)
    den = (mu_p ** 2 + mu_g ** 2 + c1) * (sigma_p ** 2 + sigma_g ** 2 + c2)
    return num / (den + 1e-8)


def compute_fmeasure(pred, gt, beta2=0.3, num_thresh=255):
    pred = pred.astype(np.float64) / 255.0 if pred.max() > 1 else pred.astype(np.float64)
    gt = (gt > 127).astype(np.float64) if gt.max() > 1 else (gt > 0.5).astype(np.float64)

    thresholds = np.linspace(0, 1, num_thresh)
    f_scores = []
    for t in thresholds:
        bin_pred = (pred >= t).astype(np.float64)
        tp = (bin_pred * gt).sum()
        fp = (bin_pred * (1 - gt)).sum()
        fn = ((1 - bin_pred) * gt).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
        f_scores.append(f)

    return float(np.max(f_scores))


def compute_emeasure(pred, gt, num_thresh=255):
    pred = pred.astype(np.float64) / 255.0 if pred.max() > 1 else pred.astype(np.float64)
    gt = (gt > 127).astype(np.float64) if gt.max() > 1 else (gt > 0.5).astype(np.float64)

    thresholds = np.linspace(0, 1, num_thresh)
    e_scores = []
    for t in thresholds:
        bin_pred = (pred >= t).astype(np.float64)
        mu_p = bin_pred.mean()
        mu_g = gt.mean()
        phi_p = bin_pred - mu_p
        phi_g = gt - mu_g
        align = (2 * phi_p * phi_g + 1e-8) / (phi_p ** 2 + phi_g ** 2 + 1e-8)
        e_score = ((align + 1) ** 2 / 4).mean()
        e_scores.append(float(e_score))

    return float(np.max(e_scores))


def compute_metrics(pred, gt):
    return {
        "MAE": compute_mae(pred, gt),
        "S": compute_smeasure(pred, gt),
        "F": compute_fmeasure(pred, gt),
        "E": compute_emeasure(pred, gt),
    }
