import torch
import scipy
import numpy as np
from models.ts2vec.ts2vec import TS2Vec
from utils.metric_common import load_metric_checkpoint, metric_checkpoint_path, save_metric_checkpoint


def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def context_fid(
    ori_data,
    generated_data,
    device=None,
    checkpoint_dir="experiments/metric_checkpoints",
    dataset_name="reference",
):
    """Reuse TS2Vec trained on the real reference and compare real/fake embeddings."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    iterations = 200 if ori_data.size <= 100000 else 600
    model = TS2Vec(
        input_dims=ori_data.shape[-1], device=device, batch_size=8,
        lr=0.001, output_dims=320, max_train_length=3000,
    )
    checkpoint_path = metric_checkpoint_path(
        checkpoint_dir, "ts2vec", ori_data, dataset_name
    )
    if not load_metric_checkpoint(model.net, checkpoint_path):
        model.fit(ori_data, n_iters=iterations, verbose=False)
        save_metric_checkpoint(model.net, checkpoint_path)

    ori_representation = model.encode(ori_data, encoding_window='full_series')
    gen_representation = model.encode(generated_data, encoding_window='full_series')
    return calculate_fid(ori_representation, gen_representation)
