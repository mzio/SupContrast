"""
ISIC Dataset
"""
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# from utils.visualize import plot_data_batch


class ISICDataset(Dataset):
    """
    ISIC Dataset
    Args:
    - root_dir (str): Where ISIC data is located, e.g. ./datasets/data/ISIC
    - split (str): 'train, 'val', 'test'
    - transform (torchvision.transforms): Image transforms
    - task_names (str[]): List of sub-target values, e.g. 'path', 'histopathology', both
    """
    img_channels = 3
    img_resolution = 224
    img_norm = {'mean': (0.71826, 0.56291, 0.52548), 
                'std': (0.16318, 0.14502, 0.17271)}
    
    def __init__(self, root_dir, split, transform=None, task_names=['patch']):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.split = split
        self.transform = transform
        self.task_names = task_names
        
        self._target_fname = 'isic_labels.csv'
        self.df = pd.read_csv(os.path.join(self.root_dir, self._target_fname))
        # Filter for train, val, or test split
        self.df = self.df[self.df['fold'] == split]
        self.df = self.df.set_index('Image Index')
        # Throw out unknowns
        self.df = self.df.loc[self.df['benign_malignant'].isin({'benign', 'malignant'})]
        
        # Superclasses
        self.superclass_names = ['benign', 'malignant']
        self.df['malignant'] = (self.df['benign_malignant'] == 'malignant').astype(int)
        self.targets = torch.tensor(self.df['malignant'].values, dtype=torch.long)
        self.n_classes = 2
        
        # Subclasses
        if self.task_names == ['patch']:
            self.subclass_names = ['benign/no_patch', 'benign/patch', 'malignant']
            self.subclass_map = {'0_0': 0, '0_1': 1, '1_0': 2}
            spurious_attribute = 'patch'
            spurious_series = self.df['patch']
        elif self.task_names == ['histopathology']:            
            self.subclass_names = ['benign/no_hist', 'benign/hist', 'malignant']
            self.subclass_map = {'0_0': 0, '0_1': 1, '1_0': 2}
            spurious_attribute = 'diagnosis_confirm_type'
            spurious_series = (self.df['diagnosis_confirm_type'] == 'histopathology').astype(int)
            raise NotImplementedError
        elif self.task_names == ['patch', 'histopathology']:
            self.subclass_names = ['benign/patch', 'benign/no_patch/no_hist', 
                                   'benign/no_patch/hist', 'malignant']
            raise NotImplementedError
#         self.n_groups = len(self.subclass_names)
#         group_array = (spurious_array * (self.n_groups / self.n_classes) + 
#                        (self.df['benign_malignant'].values == 'benign').astype(int)).astype(int)

        self.df['sub_target'] = (self.df['malignant'].astype(str) + '_' + 
                                 spurious_series.astype(str))
        group_array = self.df['sub_target'].map(self.subclass_map)
        
        self.targets_all = {'target': np.array(self.targets),
                            'spurious': np.array(spurious_series.values),
                            'group_idx': group_array,
                            'sub_target': group_array}
        self.group_labels = self.subclass_names
        
#         # Image normalization again
#         self.img_channels = img_channels
#         self.img_resolution = img_resolution
#         self.img_norm = img_norm
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.df.index[idx]))
        image = image.convert('RGB')
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)

        return (image, label, idx)
    
    
def get_transform_ISIC(augment=False, autoaugment=False):
    test_transform_list = [
        transforms.Resize(ISICDataset.img_resolution),
        transforms.CenterCrop(ISICDataset.img_resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=ISICDataset.img_norm['mean'],
                             std=ISICDataset.img_norm['std'])
    ]
    if not augment:
        return transforms.Compose(test_transform_list), transforms.Compose(test_transform_list)

    train_transform_list = [transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip()] + test_transform_list
    if autoaugment:
        print('Using AutoAugment...')
        raise NotImplementedError  # What is ImageNetPolicy
        train_transform_list.insert(4, ImageNetPolicy())
    return transforms.Compose(train_transform_list), transforms.Compose(test_transform_list)
        
        
# def load_isic(args, task_names=['patch'], train_shuffle=True, 
#               augment=False, autoaugment=False):
#     transform_train, transform_test = get_transform_ISIC(augment, autoaugment)
#     train_set = ISICDataset(root_dir=args.root_dir, split='train', 
#                            transform=transform_train, 
#                            task_names=task_names)
#     train_loader = DataLoader(train_set, batch_size=args.bs_trn,
#                               shuffle=train_shuffle, 
#                               num_workers=args.num_workers)
    
#     val_set = ISICDataset(root_dir=args.root_dir, split='val', 
#                          transform=transform_test, 
#                          task_names=task_names)
#     val_loader = DataLoader(val_set, batch_size=args.bs_val,
#                             shuffle=False, 
#                             num_workers=args.num_workers)
    
#     test_set = ISICDataset(root_dir=args.root_dir, split='test', 
#                           transform=transform_test, 
#                           task_names=task_names)
#     test_loader = DataLoader(test_set, batch_size=args.bs_val,
#                              shuffle=False, 
#                              num_workers=args.num_workers)
    
#     return (train_loader, val_loader, test_loader)

def load_isic(args, task_names=['patch'], train_shuffle=True, 
              train_transform=None, eval_transform=None):
#     transform_train, transform_test = get_transform_ISIC(augment, autoaugment)
    train_set = ISICDataset(root_dir=args.root_dir, split='train', 
                           transform=train_transform, 
                           task_names=task_names)
    train_loader = DataLoader(train_set, batch_size=args.bs_trn,
                              shuffle=train_shuffle, 
                              num_workers=args.num_workers)
    
    val_set = ISICDataset(root_dir=args.root_dir, split='val', 
                         transform=eval_transform, 
                         task_names=task_names)
    val_loader = DataLoader(val_set, batch_size=args.bs_val,
                            shuffle=False, 
                            num_workers=args.num_workers)
    
    test_set = ISICDataset(root_dir=args.root_dir, split='test', 
                          transform=eval_transform, 
                          task_names=task_names)
    test_loader = DataLoader(test_set, batch_size=args.bs_val,
                             shuffle=False, 
                             num_workers=args.num_workers)
    
    return (train_loader, val_loader, test_loader)



# Should refactor this one into dataset agnostic visualizer
def visualize_isic(dataloader, num_datapoints, title, args, save,
                   save_id, img_norm, ftype='png', target_type='group_idx'):
    # Filter for selected datapoints (in case we use SubsetRandomSampler)
    try:
        subset_indices = dataloader.sampler.indices
        targets = dataloader.dataset.targets_all[target_type][subset_indices]
        subset = True
    except AttributeError:
        targets = dataloader.dataset.targets_all[target_type]
        subset = False
    all_data_indices = []
    for class_ in np.unique(targets):
        class_indices = np.where(targets == class_)[0]
        if subset:
            class_indices = subset_indices[class_indices]
        all_data_indices.extend(class_indices[:num_datapoints])
    
    plot_data_batch([dataloader.dataset.__getitem__(ix)[0] for ix in all_data_indices],
                    mean=np.mean(img_norm['mean']), std=np.mean(img_norm['std']), 
                    nrow=8, title=title, args=args, save=save, 
                    save_id=save_id, ftype=ftype)
    
    
# Refactor for modularity
def load_dataloaders(args, task_names=['patch'], train_shuffle=True, 
                     augment=False, autoaugment=False):
    return load_isic(args, task_names, train_shuffle, 
                     augment, autoaugment)


def visualize_dataset(dataloader, num_datapoints, title, args, save,
                            save_id, ftype='png', target_type='target'):
    img_norm = {'mean': (0.71826, 0.56291, 0.52548), 
                'std': (0.16318, 0.14502, 0.17271)}
    return visualize_isic(dataloader, num_datapoints, title, 
                          args, save, save_id, img_norm, ftype, target_type)