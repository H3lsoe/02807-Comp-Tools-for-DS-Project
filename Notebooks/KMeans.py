from sklearn.preprocessing import StandardScaler
import random
import pandas as pd
import numpy as np

def kmeans(k, df, features, seed):
    
    # retrieve relevant data based on features selected
    X_scaled = initialize_data(df, features)

    # define points in feature-space
    points = X_scaled.to_numpy()

    # initialize k (random) centroids
    rng = np.random.default_rng(seed)  
    idx = rng.choice(points.shape[0], size=k, replace=False)
    centroids = points[idx]

    # initialize clustering variable mapping indices to cluster
    clustering = {i: -1 for i in range(len(points))}

    # stop criteria
    new_assignment = True

    while new_assignment:
        new_assignment = False
        # stores points in each cluster
        clusters = [[] for _ in range(k)]

        # assign points to clusters
        for i, p in enumerate(points):
            min_dist = np.inf
            p_cluster = clustering[i]

            for j, c in enumerate(centroids):
                dist_to_centroid = dist(p, c)
                if dist_to_centroid < min_dist:
                    min_dist = dist_to_centroid
                    p_cluster = j
            
            if clustering[i] != p_cluster:
                new_assignment = True
            clustering[i] = p_cluster
            clusters[p_cluster].append(i)

        # update cluster coordinates
        for i, c in enumerate(centroids):
            if len(clusters[i]) > 0:
                centroids[i] = np.mean(points[clusters[i]], axis=0)


    return clustering, centroids



    
def initialize_data(df, features):
    # ensure features are valid
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns do not exist in df: {missing_cols}")
    # extract relevant data and drop none-values
    X = df[features].copy()
    X = X.dropna()

    # -------------------------
    # 3. Standardize data
    # -------------------------
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=features,
        index=X.index 
    )

    return X_scaled

def dist(p1, p2):
    # return np.sqrt(np.sum((p1 - p2)**2))
    return np.sum((p1 - p2)**2)