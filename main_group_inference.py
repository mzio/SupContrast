from __future__ import print_function

import os
import sys
import argparse
import time
import math

import numpy as np

# import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
# Because of the following

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from networks.lenet import SupConLeNet
from losses import SupConLoss, LabeledContrastiveLoss

# New imports
## Data
from datasets.waterbirds import load_waterbirds
from datasets.celebA import load_celeba, CelebA
from datasets.isic import load_isic, get_transform_ISIC, ISICDataset

## Computing embeddings, clustering, inferring groups
from embeddings import compute_umap_embeddings, compute_embeddings
from groups import compute_group_labels


try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path', 
                                 'waterbirds', 'celebA', 'isic'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'LSpread'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    
    # MZ 9/8 -> LSpread
    parser.add_argument('--a_lc', type=float, default=1,
                        help='lc weighting term')
    parser.add_argument('--a_spread', type=float, 
                        default=0.0,
                        help='spread weighting term')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    
    # Saving
    parser.add_argument('--model_dir', type=str, 
                        default='./save/SupCon/',
                        help='Save model directory')
    parser.add_argument('--tb_dir', type=str, 
                        default='./save/SupCon/',
                        help='Tensorboard directory')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'
    opt.model_path = os.path.join(opt.model_dir,
                                  f'{opt.dataset}_models')
    opt.tb_path = os.path.join(opt.tb_dir,
                               f'{opt.dataset}_tensorboard')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)
                        
    if opt.method == 'LSpread':
        opt.model_name = f'{opt.model_name}_alc_{opt.a_lc}_aspread_{opt.a_spread}'
                        

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
        
    args_dict = vars(opt)
    print('Arguments:')
    for k, v in args_dict.items():
        print(f'- {k}: {v}')

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'waterbirds':
        mean = np.mean([0.485, 0.456, 0.406]),
        std = np.mean([0.229, 0.224, 0.225])
    elif opt.dataset == 'celebA':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'isic':
        mean = (0.71826, 0.56291, 0.52548)
        std = (0.16318, 0.14502, 0.17271)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 9/7/21 - MZ hacks (START)
    if opt.dataset == 'waterbirds':
        args = opt
        args.root_dir = '/raid/danfu/data'  # <- Change to dataset location
        args.target_name = 'waterbird_complete95'
        args.confounder_names = ['forest2water2']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.train_classes = ['landbirds', 'waterbirds']
        if args.dataset == 'waterbirds_r':
            args.train_classes = ['land', 'water']
            
        # Model
        args.arch = args.model
        args.bs_trn = args.batch_size
        args.bs_val = args.batch_size
            
        # Modified train_transform()
        # - No color jitter or grayscaling
        target_resolution = (224, 224)
        # Size?
        target_resolution = (32, 32)
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=target_resolution, 
                scale=(0.7, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
            
        train_loader, _, _ = load_waterbirds(args,
                                             train_shuffle=True,
                                             train_transform=TwoCropTransform(train_transform))
        embedding_dataloader, _, _ = load_waterbirds(args,
                                                     train_shuffle=False)
        
        return train_loader, embedding_dataloader
    elif opt.dataset == 'celebA':
        args = opt
        args.root_dir = '/raid/danfu/data/celeba'  # <- Change to dataset location
        # IMPORTANT - dataloader assumes that we have directory structure
        # in ./datasets/data/CelebA/ :
        # |-- list_attr_celeba.csv
        # |-- list_eval_partition.csv
        # |-- img_align_celeba/
        #     |-- image1.png
        #     |-- ...
        #     |-- imageN.png
        args.target_name = 'Blond_Hair'
        args.confounder_names = ['Male']
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = './images/celebA/'
        args.train_classes = ['blond', 'nonblond']
        args.val_split = 0.2
        
        args.arch = args.model
        args.bs_trn = args.batch_size
        args.bs_val = args.batch_size
        
        orig_w = 178
        orig_h = 218
        orig_min_dim = min(orig_w, orig_h)
        target_resolution = (224, 224)

        # Modified train_transform()
        # - No color jitter or grayscaling
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=orig_min_dim, scale=(0.7, 1.)),  # Added
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),  
            transforms.RandomHorizontalFlip(),  # Added
            transforms.ToTensor(),
            transforms.Normalize(mean=CelebA._normalization_stats['mean'],
                                 std=CelebA._normalization_stats['std']),
        ])
        
        # load_celeba(args, train_shuffle=True, transform=None)
        train_loader, _, _ = load_celeba(args,
                                         train_shuffle=True,
                                         train_transform=TwoCropTransform(train_transform))
        embedding_loader, _, _ = load_celeba(args,
                                             train_shuffle=False)
        
        return train_loader, embedding_loader
    
    elif opt.dataset == 'isic':
        args = opt
        args.root_dir = '/raid/danfu/data/isic_data'  # <- Change to dataset location
        args.target_name = 'benign_malignant'
        args.confounder_names = ['patch']
        args.image_mean = np.mean([0.71826, 0.56291, 0.52548])
        args.image_std = np.mean([0.16318, 0.14502, 0.17271])
        args.augment_data = False
        args.image_path = './images/isic/'
        args.train_classes = ['benign', 'malignant']
        
        args.arch = args.model
        args.bs_trn = args.batch_size
        args.bs_val = args.batch_size
        
        # No additional augmentation
        transform_list = [
            transforms.RandomResizedCrop(size=ISICDataset.img_resolution, scale=(0.7, 1.)),  # Added
            transforms.Resize(ISICDataset.img_resolution),
            transforms.CenterCrop(ISICDataset.img_resolution),
            transforms.RandomHorizontalFlip(),  # Added
            transforms.RandomVerticalFlip(),  # Added
            transforms.ToTensor(),
            transforms.Normalize(mean=ISICDataset.img_norm['mean'],
                                 std=ISICDataset.img_norm['std'])
        ]
        transform = transforms.Compose(transform_list)
        
        test_transform_list = [
            transforms.Resize(ISICDataset.img_resolution),
            transforms.CenterCrop(ISICDataset.img_resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=ISICDataset.img_norm['mean'],
                                 std=ISICDataset.img_norm['std'])
        ]
        test_transform = transforms.Compose(test_transform_list)
        
        # load_isic(args, task_names=['patch'], train_shuffle=True, 
        #           augment=False, autoaugment=False)
        train_loader, _, _ = load_isic(args,
                                       train_shuffle=True,
                                       train_transform=TwoCropTransform(transform),
                                       eval_transform=transform)
        embedding_loader, _, _ = load_isic(args,
                                           train_shuffle=False,
                                           train_transform=test_transform)
        
        return train_loader, embedding_loader
        
        
        
    # 9/7/21 - MZ hacks (END)

    elif opt.dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'path':
        train_dataset = torchvision.datasets.ImageFolder(root=opt.data_folder,
                                             transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    if opt.model == 'lenet':
        model = SupConLeNet()
    else:
        model = SupConResNet(name=opt.model)
    
    if opt.method == 'LSpread':
        criterion = LabeledContrastiveLoss(temp=opt.temp,
                                           a_lc=opt.a_lc,
                                           a_spread=opt.a_spread)
    else:
        criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.to(opt.device)
        criterion = criterion.to(opt.device)
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt, embedding_loader):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.dataset in ['waterbirds', 'isic', 'celebA']:
            images, labels, dataset_idxs = data
        else:
            images, labels = data
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'LSpread':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            
            
            # Debugging:
            if opt.trial == '42':
                # Compute group indices too
                # 1st compute embeddings
                print_str = f'-' * 5 + ' [TEST] Inferring subgroups ' + '-' * 5
                print(print_str)
                embeddings = compute_embeddings(embedding_loader, model, opt)
                # 2nd do dim reduction
                n_components = 2
                umap_seed = 42

                print(f'> Computing UMAP')
                umap_embeddings, all_indices = compute_umap_embeddings(embeddings, 
                                                                       n_components=n_components,
                                                                       seed=umap_seed)
                # Then save group predictions
                dataset = embedding_loader.dataset
                cluster_method = 'kmeans'
                n_clusters = 2
                save_dir = f'./group_predictions/{opt.dataset}'
                save_name = f'{opt.model_name}-e={epoch}-cm={cluster_method}-nc={n_clusters}-umap_nc={n_components}_s={umap_seed}.npy'

                print(f'> Clustering groups')
                pred_group_labels, prfs = compute_group_labels(umap_embeddings,
                                                               all_indices,
                                                               embedding_loader,
                                                               cluster_method,
                                                               n_clusters,
                                                               save_name,
                                                               save_dir=save_dir,
                                                               verbose=True,
                                                               norm_cost_matrix=True,
                                                               save=True,
                                                               seed=umap_seed)
                model.train()
                model = model.to(opt.device)
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader, embedding_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt, embedding_loader)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        # logger.log_value('loss', loss, epoch)
        # logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)
            
            # Compute group indices too
            # 1st compute embeddings
            print_str = f'-' * 5 + ' Inferring subgroups ' + '-' * 5
            print(print_str)
            embeddings = compute_embeddings(embedding_loader, model, opt)
            # 2nd do dim reduction
            n_components = 2
            umap_seed = 42
            
            print(f'> Computing UMAP')
            umap_embeddings, all_indices = compute_umap_embeddings(embeddings, 
                                                         n_components=n_components,
                                                         seed=umap_seed)
            # Then save group predictions
            dataset = embedding_loader.dataset
            cluster_method = 'kmeans'
            n_clusters = 2
            save_dir = f'./group_predictions/{opt.dataset}'
            save_name = f'{opt.model_name}-cm={cluster_method}-nc={n_clusters}-umap_nc={n_components}_s={umap_seed}-e={epoch}'
            
            print(f'> Clustering groups')
            pred_group_labels, prfs = compute_group_labels(umap_embeddings,
                                                           all_indices,
                                                           embedding_loader,
                                                           cluster_method,
                                                           n_clusters,
                                                           save_name,
                                                           save_dir=save_dir,
                                                           verbose=True,
                                                           norm_cost_matrix=True,
                                                           save=True,
                                                           seed=umap_seed)
            model.train()
            model = model.to(opt.device)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()
