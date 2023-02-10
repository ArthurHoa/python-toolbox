from pandas.core.frame import DataFrame
from sklearn.decomposition import PCA
from sklearn import preprocessing
import seaborn as sns
import random
import numpy as np
import matplotlib.pyplot as plt

# Available datasets in PCA/datasets
DATASET = "dog-7"

X = np.load("datasets/" + DATASET + "/X.npy").astype(float)
y_cred = np.load("datasets/" + DATASET + "/y.npy").astype(float)
y_real = np.load("datasets/" + DATASET + "/y_real.npy").astype(float)
classes = np.load("datasets/" + DATASET + "/classes.npy")

n_components = min(X.shape[0], X.shape[1])
 
# Run PCA
pca = PCA(n_components=n_components)
X = preprocessing.scale(X)
reduced = pca.fit_transform(X)

df = DataFrame(data=X)

# Append the principle components for each entry to the dataframe
for i in range(0, n_components):
    df['PC' + str(i + 1)] = reduced[:, i]

total_variance = 0
for i in range(0, n_components):
    total_variance += pca.explained_variance_ratio_[i]
    print("Variance explained for component ", i + 1, ":", total_variance)

df['Species'] = [classes[int(i)] for i in y_real]

# Show the points in terms of the first two PCs
col = ['#2ca02c', '#1f77b4', '#e377c2', '#9467bd', '#ff7f0e', '#d62728', '#8c564b', '#7f7f7f', '#bcbd22', '#17becf']
g = sns.lmplot(x='PC1',
               y='PC2',
               hue='Species',data=df,
               fit_reg=False,
               scatter=True,
               palette=col,
               legend=True,
               height=7)

#sns.move_legend(g, "upper right")
#plt.setp(g._legend.get_texts(), fontsize=13)
#plt.setp(g._legend.get_title(), fontsize=12)
plt.show()

