import torch
import numpy as np

# WARNING: AI
def KMeans(X, n_clusters, max_iter=1000):
    # move X to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if type(X) == np.ndarray:
        X = torch.from_numpy(X)
    X = X.to(device)

    n_samples, n_features = X.size()
    centers = X[torch.randperm(n_samples)[:n_clusters]].to(device)
    for _ in range(max_iter):
        dists = torch.cdist(X, centers)
        labels = torch.argmin(dists, dim=1)
        # Handle clusters with no assigned points by keeping the old center
        new_centers = torch.stack([
            X[labels == i].mean(dim=0) if (labels == i).any() else centers[i]
            for i in range(n_clusters)
        ])
        # Use allclose for floating point comparison
        if torch.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    return labels, centers