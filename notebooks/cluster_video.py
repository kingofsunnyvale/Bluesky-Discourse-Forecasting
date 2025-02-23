import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from sklearn.decomposition import PCA
from accelerate import KMeans
from tqdm import tqdm

CLUSTER_COUNT = 500
MIN_POST_FILTER = 50
FRAME_DIR = Path('../data/video_frames')
DATA_DIR = Path('../data/embeddings')
GROUP_SIZE = 2

# Prepare directories
FRAME_DIR.mkdir(exist_ok=True)

# Data loader
block_names = [f"{i:03d}.parquet" for i in range(350)]

datasets = [block_names[i:i+GROUP_SIZE] for i in range(0, len(block_names), GROUP_SIZE)]

def flatten(column):
    return np.array([np.array(x) for x in column])

centrioids = None

for dataset_number, dataset in enumerate(tqdm(datasets, desc="Processing datasets")):
    print("Processing dataset", dataset_number)

    dataset = pd.concat([pd.read_parquet(DATA_DIR / block) for block in dataset])

    embedding_matrix = torch.from_numpy(flatten(dataset['embeddings'])).to('cuda:0')

    class_labels, class_means = KMeans(embedding_matrix, CLUSTER_COUNT)

    dataset["cluster"] = class_labels.cpu().numpy()

    class_mse = np.zeros(CLUSTER_COUNT)
    for i in range(CLUSTER_COUNT):
        class_mse[i] = torch.mean((embedding_matrix[class_labels == i] - class_means[i])**2).cpu().numpy()

    class_sizes = np.array([torch.sum(class_labels == i).cpu().numpy() for i in range(CLUSTER_COUNT)])

    # filter out clusters with less than min_posts
    if centrioids is None:
        large_clusters = torch.from_numpy(np.array(np.where(class_sizes > MIN_POST_FILTER))).to('cuda:0')

        embeddings = embedding_matrix[torch.isin(class_labels, large_clusters)]
        class_labels = class_labels[torch.isin(class_labels, large_clusters)]

        print(f"Reduced to {embeddings.shape[0]} tweets in {len(large_clusters[0])} clusters")

        # lowest MSE clusters
        class_MSE = torch.zeros(CLUSTER_COUNT) + torch.inf
        MSE = torch.mean((embeddings - class_means[class_labels])**2, dim=1)
        for i in large_clusters.cpu().numpy().flatten():
            class_MSE[i] = torch.mean(MSE[class_labels == i])
    else:
        embeddings = embedding_matrix

    # end of GPU
    class_means = class_means.cpu().numpy()
    class_labels = class_labels.cpu().numpy()
    embeddings = embeddings.cpu().numpy()

    if centrioids is None:
        clusters_ranked = np.argsort(class_MSE)
        best_clusters = clusters_ranked[0:10]
    else:
        # for each point in centroid, find the closest class_mean
        dists = np.zeros((centrioids.shape[0], class_means.shape[0]))
        for i in range(centrioids.shape[0]):
            for j in range(class_means.shape[0]):
                dists[i, j] = np.linalg.norm(centrioids[i] - class_means[j])
        best_clusters = np.argmin(dists, axis=1)

    centrioids = class_means[best_clusters]

    print(f"Best clusters: {best_clusters}")

    # PCA for the top M clusters
    pca = PCA(n_components=2)
    pca.fit(class_means)
    pca_mean = pca.transform(class_means)
    pca_datapoints = pca.transform(embeddings)

    cmap = plt.get_cmap('jet')

    plt.figure()
    for i in range(len(best_clusters)):
        plt.scatter(pca_mean[best_clusters[i], 0], pca_mean[best_clusters[i], 1], color=cmap(i/len(best_clusters)), marker='x')

        pca_class_datapoints = pca_datapoints[class_labels == best_clusters[i]]
        print(f"Class {best_clusters[i]}: {pca_class_datapoints.shape[0]} tweets")
        plt.scatter(pca_class_datapoints[:, 0], pca_class_datapoints[:, 1], color=cmap(i/len(best_clusters)), alpha=0.1)
        plt.xlim(-1/2, 3/4)
        plt.ylim(-1/2, 3/4)
    plt.savefig(FRAME_DIR / f"frame_{dataset_number:03d}.png")
