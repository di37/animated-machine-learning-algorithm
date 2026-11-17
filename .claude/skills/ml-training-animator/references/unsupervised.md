# Unsupervised learning: clustering, dimensionality reduction, density and anomalies

Read with `SKILL.md`; its contract, truthfulness rules, branding and pipeline all apply. There are no labels
to learn from, so the story is **structure found in unlabelled data**, and "inference" is what the fitted
model says about a new point.

## Default story (adapt the lesson's ten chapters)

title → **data** (points appear unlabelled, one neutral colour) → **initial state** (random centroids, a
random embedding, every point its own cluster) → **iterations** (real steps, with the objective falling or
rising alongside) → **final structure** → **new point** (assigned, projected or scored) → **choosing the
setting** (k, eps, perplexity: a real sweep) → takeaway and limitation.

Default to **2 features** so the data itself is the picture. With more, cluster in the full space and draw a
**stated projection** (a PCA fitted once, on the training rows), and say the clustering did not use it.

## Real states: how to capture them

| Algorithm | Real training states | Inference on a new point |
| --- | --- | --- |
| k-means | Lloyd iterations: run the assign/update loop yourself from sklearn's initial centroids, or `KMeans(init=centroids, n_init=1, max_iter=1)` once per step. Record centroids, labels and inertia per step | `predict`: nearest centroid, with the distances drawn |
| Mini-batch k-means | `partial_fit` per batch | as k-means |
| Gaussian mixture | EM steps: `GaussianMixture(warm_start=True, max_iter=1)`, refitted once per step (expect a convergence warning). Record means, covariances (draw ellipses), weights, `lower_bound_` | `predict_proba`: responsibilities per component, then the label |
| DBSCAN / HDBSCAN | One pass, no iterations. Show an **honest construction**: neighbour counts give core points, then cluster expansion from core points (your own BFS over the same neighbourhoods). Check that the final labels equal sklearn's | No `predict`. Say so; if you show a rule (nearest core point within eps), label it as your rule |
| Agglomerative | The real merge sequence: `children_` and `distances_` (`compute_distances=True`); grow the dendrogram and the merged groups together | No `predict`; cut the tree at a stated height |
| PCA | Closed form: covariance, eigenvectors as arrows, explained variance. `IncrementalPCA.partial_fit` batches are real steps; power iteration only as a labelled computation | `transform`: drop the point onto the components, then reconstruct it |
| t-SNE | Positions *are* the output. sklearn's `TSNE` exposes no iterations and cannot place new points; use openTSNE (callbacks every n iterations, KL divergence) to record real steps | openTSNE `transform`; never call an sklearn t-SNE map an embedding of a new point |
| UMAP | `n_epochs` as a list keeps each intermediate layout in `embedding_list_` | `transform` |
| Isolation forest | Trees added with `warm_start=True` and growing `n_estimators` | `score_samples` / `decision_function`, with the threshold |
| Autoencoder | Deep learning: see `deep-learning.md` | Encode, show the latent point, decode |

## Metrics without labels

Use the algorithm's own objective (inertia, log-likelihood or `lower_bound_`, KL divergence, reconstruction
error, explained variance) plus one external check (silhouette on a stated sample, Davies–Bouldin,
trustworthiness for embeddings). If ground-truth labels exist, show them **only in a reference panel marked
"never seen by the model"**, and compare with adjusted Rand or mutual information, never accuracy.

## Visual grammar and pitfalls

- **Label switching:** cluster ids are arbitrary and can permute between steps, seeds or k. Match each step's
  clusters to the previous step's (Hungarian matching on centroid distance) before colouring. A colour flip
  is not the model changing its mind.
- **Points stay put in data space.** Only centroids, ellipses, boundaries and memberships move. In an
  embedding the positions *are* the output; fix the axes over every recorded step (or rescale every step the
  same way) and say what moves.
- **Seeds and restarts:** k-means and GMM depend on initialisation. Show the seed. When `n_init > 1`, the
  video shows one run; say which (the best inertia) and that the others were discarded.
- **Choosing k, eps or perplexity:** sweep real values (elbow, silhouette, BIC) and mark the chosen one. Never
  imply the algorithm chose k itself unless it did (DBSCAN, HDBSCAN).
- **Noise:** draw DBSCAN noise points as their own marker, not as a cluster.
- **Scale:** standardise with training-row statistics, before clustering; distances depend on it, and the
  video should say it was done.
- **t-SNE and UMAP:** distances between far-apart clusters and cluster sizes do not mean much. Say so in the
  takeaway, and do not draw conclusions the layout cannot support.

## Takeaways and limitations worth stating

k-means assumes round, similar-sized clusters and needs k; GMM fits ellipses and can overfit with full
covariances; DBSCAN needs one density level; agglomerative clustering's result depends on the linkage; PCA
finds only linear structure; t-SNE and UMAP preserve neighbourhoods, not global distances.
