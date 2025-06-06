import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

features = torch.load("misc/facade_anchor_features/facade_features.pt", map_location="cpu")
x1, x2 = features["left"].detach().numpy(), features["right"].detach().numpy()
x = np.concatenate([x1, x2], axis=0)
model = TSNE(n_components=2)
reduced = model.fit_transform(x)

x1_reduced, x2_reduced = reduced[: len(x1)], reduced[len(x1) :]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(x1_reduced[:, 0], x1_reduced[:, 1], c="red", label="left facade")
ax.scatter(x2_reduced[:, 0], x2_reduced[:, 1], c="blue", label="right facade")
ax.legend()
ax.set_title("t-SNE")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.grid(True)
fig.savefig("tsne.png")
plt.close(fig)
