import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

train_path = 'data/features/processed/train_data_processed.csv'
test_path = 'data/features/processed/test_data_processed.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

selected_fruits = ['Apple', 'Banana', 'Avocado', 'Coco']
train_data = train_data[train_data['label'].isin(selected_fruits)]
test_data = test_data[test_data['label'].isin(selected_fruits)]

X_train = train_data.drop(columns=['label', 'label_encoded'])
y_train = train_data['label']
X_test = test_data.drop(columns=['label', 'label_encoded'])
y_test = test_data['label']

print(f'Filtered dataset - Train samples: {len(y_train)}, Test samples: {len(y_test)}')
print(f'Class distribution (Train):')
print(y_train.value_counts())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

cumsum = np.cumsum(pca.explained_variance_ratio_)
n_components_95 = np.argmax(cumsum >= 0.95) + 1
n_components_99 = np.argmax(cumsum >= 0.99) + 1

print(f'Total features: {X_train.shape[1]}')
print(f'Components for 95% variance: {n_components_95}')
print(f'Components for 99% variance: {n_components_99}')
print(f'Variance explained by first 2 components: {pca.explained_variance_ratio_[:2].sum():.4f}')
print(f'Variance explained by first 3 components: {pca.explained_variance_ratio_[:3].sum():.4f}')

pca_2d = PCA(n_components=2)
X_train_pca_2d = pca_2d.fit_transform(X_train_scaled)

unique_labels = np.unique(y_train)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

# t-SNE 2D
print('\nCalculating t-SNE 2D...')
tsne_2d = TSNE(n_components=2, random_state=42, perplexity=30)
X_train_tsne_2d = tsne_2d.fit_transform(X_train_scaled)

# t-SNE 3D
print('Calculating t-SNE 3D...')
tsne_3d = TSNE(n_components=3, random_state=42, perplexity=30)
X_train_tsne_3d = tsne_3d.fit_transform(X_train_scaled)

# UMAP 2D
print('Calculating UMAP 2D...')
umap_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_train_umap_2d = umap_2d.fit_transform(X_train_scaled)

# UMAP 3D
print('Calculating UMAP 3D...')
umap_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
X_train_umap_3d = umap_3d.fit_transform(X_train_scaled)

# PCA Figure
fig1 = plt.figure(figsize=(14, 6))
fig1.suptitle('PCA - Principal Component Analysis', fontsize=16, fontweight='bold')

# PCA 2D
ax1 = fig1.add_subplot(1, 2, 1)
for i, label in enumerate(unique_labels):
    mask = y_train == label
    ax1.scatter(X_train_pca_2d[mask, 0], X_train_pca_2d[mask, 1], 
                c=[colors[i]], label=label, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%})')
ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%})')
ax1.set_title('Train Set - PCA 2D')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

pca_3d = PCA(n_components=3)
X_train_pca_3d = pca_3d.fit_transform(X_train_scaled)

# PCA 3D
ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
for i, label in enumerate(unique_labels):
    mask = y_train == label
    ax2.scatter(X_train_pca_3d[mask, 0], X_train_pca_3d[mask, 1], X_train_pca_3d[mask, 2],
                c=[colors[i]], label=label, s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%})')
ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%})')
ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%})')
ax2.set_title('Train Set - PCA 3D')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()

# t-SNE Figure
fig2 = plt.figure(figsize=(14, 6))
fig2.suptitle('t-SNE - t-Distributed Stochastic Neighbor Embedding', fontsize=16, fontweight='bold')

# t-SNE 2D
ax3 = fig2.add_subplot(1, 2, 1)
for i, label in enumerate(unique_labels):
    mask = y_train == label
    ax3.scatter(X_train_tsne_2d[mask, 0], X_train_tsne_2d[mask, 1], 
                c=[colors[i]], label=label, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax3.set_xlabel('t-SNE 1')
ax3.set_ylabel('t-SNE 2')
ax3.set_title('Train Set - t-SNE 2D')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax3.grid(True, alpha=0.3)

# t-SNE 3D
ax4 = fig2.add_subplot(1, 2, 2, projection='3d')
for i, label in enumerate(unique_labels):
    mask = y_train == label
    ax4.scatter(X_train_tsne_3d[mask, 0], X_train_tsne_3d[mask, 1], X_train_tsne_3d[mask, 2],
                c=[colors[i]], label=label, s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
ax4.set_xlabel('t-SNE 1')
ax4.set_ylabel('t-SNE 2')
ax4.set_zlabel('t-SNE 3')
ax4.set_title('Train Set - t-SNE 3D')
ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()

# UMAP Figure
fig3 = plt.figure(figsize=(14, 6))
fig3.suptitle('UMAP - Uniform Manifold Approximation and Projection', fontsize=16, fontweight='bold')

# UMAP 2D
ax5 = fig3.add_subplot(1, 2, 1)
for i, label in enumerate(unique_labels):
    mask = y_train == label
    ax5.scatter(X_train_umap_2d[mask, 0], X_train_umap_2d[mask, 1], 
                c=[colors[i]], label=label, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax5.set_xlabel('UMAP 1')
ax5.set_ylabel('UMAP 2')
ax5.set_title('Train Set - UMAP 2D')
ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax5.grid(True, alpha=0.3)

# UMAP 3D
ax6 = fig3.add_subplot(1, 2, 2, projection='3d')
for i, label in enumerate(unique_labels):
    mask = y_train == label
    ax6.scatter(X_train_umap_3d[mask, 0], X_train_umap_3d[mask, 1], X_train_umap_3d[mask, 2],
                c=[colors[i]], label=label, s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
ax6.set_xlabel('UMAP 1')
ax6.set_ylabel('UMAP 2')
ax6.set_zlabel('UMAP 3')
ax6.set_title('Train Set - UMAP 3D')
ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()

print(f'\nTrain set shape after PCA (2D): {X_train_pca_2d.shape}')
print(f'Train set shape after PCA (3D): {X_train_pca_3d.shape}')
print(f'Train set shape after t-SNE (2D): {X_train_tsne_2d.shape}')
print(f'Train set shape after t-SNE (3D): {X_train_tsne_3d.shape}')
print(f'Train set shape after UMAP (2D): {X_train_umap_2d.shape}')
print(f'Train set shape after UMAP (3D): {X_train_umap_3d.shape}')
