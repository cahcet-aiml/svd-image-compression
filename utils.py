import numpy as np
from skimage.metrics import structural_similarity as ssim

def compress_channel(channel, threshold=0.95):
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)

    for k in range(1, len(S)):
        S_k = np.diag(S[:k])
        compressed = U[:, :k] @ S_k @ Vt[:k, :]
        
        score = ssim(channel, compressed, data_range=channel.max() - channel.min())
        
        if score >= threshold:
            return compressed, k, score

    return channel, len(S), 1.0

def compress_image_auto_k(img, threshold=0.95):
    channels = []
    ks = []
    scores = []

    for i in range(3):
        comp, k, score = compress_channel(img[:, :, i], threshold)
        channels.append(comp)
        ks.append(k)
        scores.append(score)

    return np.stack(channels, axis=2), int(sum(ks)/3), sum(scores)/3
