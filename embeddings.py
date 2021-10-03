"""
Methods to compute embeddings / dimensionality-reduced components from trained models.
- Computed components are then clustered by class to infer groups
"""
import torch
import umap
import numpy as np

from tqdm import tqdm

from networks.resnet_big import SupConResNet


# --------------------------------------------------
# Methods to compute embeddings from model
# --------------------------------------------------

def compute_embeddings_from_model(dataloader, checkpoint_path, args):
    """
    Computes embeddings from last hidden layer of a ResNet50 backbone contrastive model
    Args:
    - checkpoint_path (str): path to checkpoint
    Returns: 
    - embeddings (np.array)
     
    Currently only supports contrastive ResNet-50 bc hacks
    """
    model = SupConResNet(checkpoint_path)  # 'resnet50'
    model = load_checkpoint(checkpoint_path)
    embeddings = compute_embeddings(dataloader, model, args)
    return embeddings


def load_checkpoint(checkpoint_path):
    """
    Load a previously trained model checkpoint 
    Args:
    - checkpoint_path (str): path to checkpoint
    """
    model = SupConResNet('resnet50')
    ckpt = torch.load(checkpoint_path, 
                      map_location='cpu')
    state_dict = ckpt['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v
    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    return model


def compute_embeddings(dataloader, model, args):
    """
    Actually compute the embeddings
    """
    model.eval()
    model.to(args.device)
    
    all_embeddings = []
    
    with torch.no_grad():
        for ix, data in enumerate(tqdm(dataloader, desc='> Computing embeddings')):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            embeddings = model.encoder(inputs)
            embeddings = embeddings.to(torch.device('cpu')).numpy()
            all_embeddings.append(embeddings)

            inputs = inputs.to(torch.device('cpu'))
            labels = labels.to(torch.device('cpu'))

        model.to(torch.device('cpu'))
    return np.concatenate(all_embeddings)


# --------------------------------------------------
# Methods to compute UMAP representations from model embeddings
# --------------------------------------------------

def compute_umap_embeddings(embeddings, 
                            n_components=2,
                            seed=42):
    """
    Args:
    - embeddings (ndarray): Input embeddings (num_embeddings x embedding_dim)
    - n_components (int): Number of dimensions to reduce
    - seed (int): Random reproducibility thing
    Returns:
    - umap_embedding (ndarray): (num_embeddings x n_components)
    - indices (ndarray): [0, 1, 2, .... , num_embeddings]
    """
    indices = np.arange(len(embeddings))
    umap_embedding = umap.UMAP(random_state=seed,
                               n_components=n_components).fit_transform(embeddings)
    return umap_embedding, indices