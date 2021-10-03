"""
Methods to compute group predictions given (dimensionality-reduced embeddings)
"""
import numpy as np

from os.path import join

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support

from scipy.optimize import linear_sum_assignment


def get_indices_by_class(dataset):
    indices = []
    for c in range(dataset.n_classes):
        indices.append(
            np.where(dataset.targets_all['target'] == c)[0]
        )
    return indices


def get_indices_by_groups(dataset):
    """
    Only use this to see F1-scores for how well we can recover the subgroups
    """
    indices = []
    for g in range(len(dataset.group_labels)):
        indices.append(
            np.where(dataset.targets_all['group_idx'] == g)[0]
        )
    return indices


def compute_clusters(embeddings, 
                     cluster_method='kmeans',
                     num_clusters=2, 
                     indices=None,
                     label_type='target',
                     dataloader=None,
                     seed=42):
    if cluster_method == 'kmeans':
        clusterer = KMeans(n_clusters=num_clusters,
                           random_state=seed,
                           n_init=10)
        cluster_labels = clusterer.fit_predict(embeddings)
        means = clusterer.cluster_centers_
    elif cluster_method == 'gmm':
        clusterer = GaussianMixture(n_components=num_clusters,
                                    random_state=seed,
                                    n_init=10)
        cluster_labels = clusterer.fit_predict(embeddings)
        means = clusterer.means_
#     cluster_labels, cluster_correct = compute_cluster_assignment(cluster_labels, 
#                                                                  dataloader,
#                                                                  indices,
#                                                                  label_type)
    return cluster_labels, None  # , cluster_correct


def compute_group_labels(embeddings, all_indices, dataloader, cluster_method='kmeans', n_clusters=2, 
                         save_name=None, ix=None, save=True, norm_cost_matrix=True, 
                         save_dir='./group_predictions', verbose=False, seed=42):
    """
    Compute group labels given embeddings
    - Will also report precision, recall, f1-score for the groups (assuming best-assignment)
      - Note this requires referencing the true group labels, and then mapping arbirtrary group indices generated by the clustering to an assignment that best matches these true group labels
      - But GDRO sees any group index assignment as the same. 
        - There is no advantage with mapping the arbitrary group indices here
    """
    all_cluster_labels = []
    all_indices_by_pred_groups = []
    
    ref_indices_by_class = get_indices_by_class(dataloader.dataset)
    ref_indices_by_group = get_indices_by_groups(dataloader.dataset)

    for cix, class_indices in enumerate(ref_indices_by_class):
        cluster_labels, cluster_correct = compute_clusters(embeddings[class_indices],
                                                           cluster_method,
                                                           n_clusters,
                                                           indices=all_indices[class_indices],
                                                           dataloader=dataloader,
                                                           seed=seed)
        for c in np.unique(cluster_labels):
            all_indices_by_pred_groups.append(class_indices[np.where(cluster_labels == c)[0]])
            
    pred_group_labels = np.zeros(len(all_indices))
    for g, indices in enumerate(all_indices_by_pred_groups):
        for i in indices:
            pred_group_labels[i] = g
    # pred_group_labels = np.concatenate(pred_group_labels)
    
    if save:
        save_path = join(save_dir, f'pred_groups_{save_name}.npy')
        with open(save_path, 'wb') as f:
            np.save(f, pred_group_labels)
            print(f'Saved group predictions to {save_path}')
            
    # For now, don't compute the F1-score
    return pred_group_labels, None

    # Compute mapping to report precision, recall, f1-score
    cost_matrix = np.zeros((len(all_indices_by_pred_groups), len(ref_indices_by_group)))
    cost_matrix_normed = np.zeros((len(all_indices_by_pred_groups), len(ref_indices_by_group)))
    for pix, pred_group_indices in enumerate(all_indices_by_pred_groups):
        for gix, group_indices in enumerate(ref_indices_by_group):
            intersection_counts = np.intersect1d(pred_group_indices, group_indices).shape[0]
            cost = -1 * intersection_counts
            cost_normed = 1 - (intersection_counts / len(group_indices))
            output = f'{pix} {gix} {intersection_counts:4d} {len(group_indices):4d} {intersection_counts / len(group_indices):<.3f}'
            if verbose:
                print(output)
            # Saving cost matrix
            cost_matrix[pix][gix] = cost
            cost_matrix_normed[pix][gix] = cost_normed
        if verbose:
            print('')
        
    if norm_cost_matrix:
        cost_matrix = cost_matrix_normed
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    print(f'Hungarian assignment: {col_ind}')

    dataset = dataloader.dataset
    pred_group_labels = np.zeros(len(dataloader.dataset.targets_all['target']))
    for pix, pred_group_indices in enumerate(all_indices_by_pred_groups):
        pred_group_labels[pred_group_indices] = col_ind[pix]
    acc = (pred_group_labels == dataset.targets_all['group_idx']).sum() / len(pred_group_labels)
    print(f'Acc: {acc * 100:<.3f}%')
    print('Precision, Recall, F1-Score (%)')
    print('- Average by:')
#     prf = precision_recall_fscore_support(dataset.targets_all['group_idx'], pred_group_labels)
#     print(f'- none: {prf}')
    micro_prf = precision_recall_fscore_support(dataset.targets_all['group_idx'], pred_group_labels, average='micro')
    prf = ' '.join([f'{m * 100:<.3f}' for m in micro_prf[:3]])
    print(f' - micro:    {prf}')
    macro_prf = precision_recall_fscore_support(dataset.targets_all['group_idx'], pred_group_labels, average='macro')
    prf = ' '.join([f'{m * 100:<.3f}' for m in macro_prf[:3]])
    print(f' - macro:    {prf}')
    weighted_prf = precision_recall_fscore_support(dataset.targets_all['group_idx'], pred_group_labels, average='weighted')
    prf = ' '.join([f'{m * 100:<.3f}' for m in weighted_prf[:3]])
    print(f' - weighted: {prf}')
    
    if save:
        save_path = join(save_dir, f'pred_groups_{save_name}.npy')
        with open(save_path, 'wb') as f:
            np.save(f, pred_group_labels)
            print(f'Saved group predictions to {save_path}')
    
    return pred_group_labels, (micro_prf, macro_prf, weighted_prf)