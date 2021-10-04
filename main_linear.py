from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn

# from main_ce import set_loader
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import SupConResNet, LinearClassifier

# New imports
## Data
from datasets.waterbirds import load_waterbirds
from datasets.celebA import load_celeba, CelebA
from datasets.isic import load_isic, get_transform_ISIC, ISICDataset

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
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'waterbirds'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
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
            
    opt.topk = (1, 5)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'waterbirds':
        opt.n_cls = 2
        opt.topk = (1, 1)  # HACK
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

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
        args.root_dir = '../slice-and-dice-smol/datasets/data/Waterbirds/'  # <- Change to dataset location
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
        
        scale = 1 # 256.0 / target_resolution[0]
        
        eval_transform = transforms.Compose([
            transforms.Resize(
                size=target_resolution),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            normalize,
        ])
            
        train_loader, val_loader, test_loader = load_waterbirds(args,
                                             train_shuffle=True,
                                             train_transform=train_transform,
                                             eval_transform=eval_transform)
        
        return train_loader, test_loader
    elif opt.dataset == 'celebA':
        args = opt
        args.root_dir = '/home/danfu/data'  # <- Change to dataset location
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
        
        eval_transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),  
            transforms.ToTensor(),
            transforms.Normalize(mean=CelebA._normalization_stats['mean'],
                                 std=CelebA._normalization_stats['std']),
        ])
        
        # load_celeba(args, train_shuffle=True, transform=None)
        train_loader, val_loader, test_loader = load_celeba(args,
                                         train_shuffle=True,
                                         train_transform=train_transform,
                                         eval_transform=eval_transform)
        
        return train_loader, test_loader
    
    elif opt.dataset == 'isic':
        args = opt
        args.root_dir = '../slice-and-dice-smol/datasets/data/ISIC/'  # <- Change to dataset location
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
        train_loader, val_loader, test_loader = load_isic(args,
                                       train_shuffle=True,
                                       train_transform=transform,
                                       eval_transform=transform)
        
        return train_loader, test_loader
        
        
        
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
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.dataset == 'waterbirds':
            images, labels, dataset_idxs = data
        else:
            images, labels = data
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=opt.topk)
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, data in enumerate(val_loader):
            if opt.dataset == 'waterbirds':
                images, labels, dataset_idxs = data
            else:
                images, labels = data
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=opt.topk)
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        if val_acc > best_acc:
            best_acc = val_acc

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
