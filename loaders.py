import os
import pickle
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Subset(data.Dataset):
    def __init__(self, dataset, indices=None):
        """
        Subset of dataset given by indices.
        """
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices

        if self.indices is None:
            self.n_samples = len(self.dataset)
        else:
            self.n_samples = len(self.indices)
            assert self.n_samples >= 0 and \
                self.n_samples <= len(self.dataset), \
                "length of {} incompatible with dataset of size {}"\
                .format(self.n_samples, len(self.dataset))

    def __getitem__(self, idx):
        if torch.is_tensor(idx) and idx.dim():
            res = [self[iidx] for iidx in idx]
            return torch.stack([x[0] for x in res]), torch.LongTensor([x[1] for x in res])
        if self.indices is None:
            return self.dataset[idx]
        else:
            return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.n_samples


def random_subsets(subset_sizes, n_total, seed=None, replace=False):
    """
    Return subsets of indices, with sizes given by the iterable
    subset_sizes, drawn from {0, ..., n_total - 1}
    Subsets may be distinct or not according to the replace option.
    Optional seed for deterministic draw.
    """
    # save current random state
    state = np.random.get_state()
    sum_sizes = sum(subset_sizes)
    assert sum_sizes <= n_total

    np.random.seed(seed)

    total_subset = np.random.choice(n_total, size=sum_sizes,
                                    replace=replace)
    perm = np.random.permutation(total_subset)
    res = []
    start = 0
    for size in subset_sizes:
        res.append(perm[start: start + size])
        start += size
    # restore initial random state
    np.random.set_state(state)
    return res



def get_data_loaders(args):
    print('Dataset: \t {}'.format(args.dataset.upper()))

    #for k in ('train_size', 'val_size', 'test_size'):
        #if args.__dict__[k] is None:
            #args.__dict__.pop(k)

    if args.dataset == 'mnist':
        loader_train, loader_val, loader_test = loaders_mnist(**vars(args))
    elif 'cifar' in args.dataset:
        loader_train, loader_val, loader_test = loaders_cifar(**vars(args))
    elif 'svhn' in args.dataset:
        loader_train, loader_val, loader_test = loaders_svhn(**vars(args))
    elif args.dataset == 'imagenet':
        loader_train, loader_val, loader_test = loaders_imagenet(**vars(args))
    elif args.dataset == 'tiny_imagenet':
        loader_train, loader_val, loader_test = loaders_tiny_imagenet(**vars(args))
    else:
        raise NotImplementedError

    args.train_size = len(loader_train.dataset)
    args.val_size = len(loader_val.dataset)
    args.test_size = len(loader_test.dataset)
    print('train size: ', args.train_size,'val size: ', args.val_size, 'test size: ', args.test_size)

    return loader_train, loader_val, loader_test

class IndexedDataset(data.Dataset):
    def __init__(self, dataset):
        self._dataset = dataset
    def __getitem__(self, idx):
        return idx, self._dataset[idx]
    def __len__(self):
        return self._dataset.__len__()

def create_loaders(dataset_train, dataset_val, dataset_test,
                   train_size, val_size, test_size, batch_size, test_batch_size,
                   cuda, n_classes, num_workers, split=True, eq_class=False):

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}

    if split:
        train_indices, val_indices = random_subsets((train_size, val_size),
                                                    len(dataset_train),
                                                    seed=1234)
    else:
        train_size = train_size if train_size is not None else len(dataset_train)
        train_indices, = random_subsets((train_size,),
                                        len(dataset_train),
                                        seed=1234)
        val_size = val_size if val_size is not None else len(dataset_val)
        val_indices, = random_subsets((val_size,),
                                      len(dataset_val),
                                      seed=1234)

    test_size = test_size if test_size is not None else len(dataset_test)
    test_indices, = random_subsets((test_size,),
                                   len(dataset_test),
                                   seed=1234)

    dataset_train = Subset(dataset_train, train_indices)
    dataset_val = Subset(dataset_val, val_indices)
    dataset_test = Subset(dataset_test, test_indices)

    print('Dataset sizes: \t train: {} \t val: {} \t test: {}'
          .format(len(dataset_train), len(dataset_val), len(dataset_test)))
    print('Batch size: \t {}'.format(batch_size))

    '''
    if eq_class:
        assert batch_size % n_classes == 0
        balanced_batch_sampler = BalancedBatchSampler(dataset_train,
                                                      n_classes, batch_size//n_classes)
        train_loader = data.DataLoader(dataset_train,
                                       batch_sampler=balanced_batch_sampler,
                                       **kwargs)
    else:
        dataset_train = IndexedDataset(dataset_train)
    '''

    train_loader = data.DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True, **kwargs)

    val_loader = data.DataLoader(dataset_val,
                                 batch_size=test_batch_size,
                                 shuffle=False, **kwargs)

    test_loader = data.DataLoader(dataset_test,
                                  batch_size=test_batch_size,
                                  shuffle=False, **kwargs)

    train_loader.tag = 'train'
    val_loader.tag = 'val'
    test_loader.tag = 'test'

    return train_loader, val_loader, test_loader

def loaders_mnist(dataset, batch_size=64, cuda=0,
                  train_size=50000, val_size=10000, test_size=10000,
                  test_batch_size=1000, augment=False, **kwargs):

    assert dataset == 'mnist'
    root = '{}/{}'.format(os.environ['VISION_DATA'], dataset)

    # Data loading code
    normalize = transforms.Normalize(mean=(0.1307,),
                                     std=(0.3081,))

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    # define two datasets in order to have different transforms
    # on training and validation
    dataset_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    dataset_val = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST(root=root, train=False, download=True, transform=transform)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size=batch_size,
                          test_batch_size=test_batch_size,
                          cuda=cuda, n_classes=10, num_workers=0)

def loaders_cifar(dataset, batch_size, cuda, n_classes,
                  train_size=45000, augment=True, val_size=5000, test_size=10000,
                  test_batch_size=128, **kwargs):

    assert dataset in ('cifar10', 'cifar100')

    if dataset == 'cifar10':
        dir_ = 'CIFAR10'
    else: 
        dir_ = 'CIFAR100'

    root = '{}/{}'.format(os.environ['VISION_DATA'], dir_)

    # Data loading code
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    if augment:
        print('Using data augmentation')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        print('Not using data augmentation')
        transform_train = transform_test

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    dataset_train = dataset(root=root, train=True, download=True,
                            transform=transform_train)
    dataset_val = dataset(root=root, train=True,
                          transform=transform_test)
    dataset_test = dataset(root=root, train=False, download=True,
                           transform=transform_test)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=0)

def loaders_svhn(dataset, batch_size, cuda,
                 n_classes, eq_class, train_size=None, augment=False, val_size=6000, test_size=26032,
                 test_batch_size=1000, **kwargs):

    assert 'svhn' in dataset

    root = '{}/svhn'.format(os.environ['VISION_DATA'])
    dataset_name = dataset

    # Data loading code
    transform_test = transforms.Compose([
        transforms.ToTensor()])

    if augment:
        assert ValueError("Should not be using data augmentation on SVHN")
    else:
        print('Not using data augmentation')
        transform_train = transform_test

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset = datasets.SVHN
    if dataset_name == 'svhn':
        train_size = 73257 - val_size if train_size is None else train_size
        dataset_train = dataset(root=root, split='train', download=True,
                                transform=transform_train)
        dataset_val = dataset(root=root, split='train',
                              transform=transform_test)
        split = True
    elif dataset_name == 'svhn-extra':
        # manual train-val split within difficult examples (i.e. not from extra data)
        train_indices, val_indices = random_subsets((73257 - val_size, val_size), 73257, seed=1234)

        # not-extra data: split betwwen train and val
        dataset_train_reduced = Subset(dataset(root=root, split='train', transform=transform_train, download=True), train_indices)
        dataset_val = Subset(dataset(root=root, split='train', transform=transform_test), val_indices)

        # add extra data to train
        dataset_train = data.ConcatDataset((dataset_train_reduced, dataset(root=root, split='extra', transform=transform_train, download=True)))
        split = False
    else:
        raise ValueError
    dataset_test = dataset(root=root, split='test', download=True,
                           transform=transform_test)

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=4, split=split)

def loaders_imagenet(dataset, batch_size, cuda, n_classes,
                                          train_size=1231166, augment=True, val_size=50000,
                                          test_size=50000, test_batch_size=256, topk=None, noise=False,
                                          multiple_crops=False, data_root=None, **kwargs):
    assert dataset == 'imagenet'
    data_root = data_root if data_root is not None else os.environ['VISION_DATA']
    root = '{}/imagenet/'.format(data_root)
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    testdir = os.path.join(root, 'test')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize])

    if augment:
        transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        Lighting(0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    else:
        transform_train = transform_test

    dataset_train = datasets.ImageFolder(traindir, transform_train)
    dataset_val = datasets.ImageFolder(valdir, transform_test)
    dataset_test = datasets.ImageFolder(valdir, transform_test)
    #dataset_test = datasets.ImageFolder(testdir, transform_test)
    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=8, split=False)


def loaders_imagenet_old(dataset, batch_size, cuda, n_classes,
                                          train_size=1231166, augment=True, val_size=50000,
                                          test_size=50000, test_batch_size=256, topk=None, noise=False,
                                          multiple_crops=False, data_root=None, **kwargs):
    assert dataset == 'imagenet'
    data_root = data_root if data_root is not None else os.environ['VISION_DATA']
    root = '{}/imagenet/'.format(data_root)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    testdir = os.path.join(root, 'test')
    normalize = transforms.Normalize(mean=mean, std=std)
    if multiple_crops:
        print('Using multiple crops')
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            lambda x: [normalize(transforms.functional.to_tensor(img)) for img in x]])
    else:
        transform_test = transforms.Compose([
            # transforms.Resize(256),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(224, pad_if_needed=True),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(
                            # brightness=0.4,
                            # contrast=0.4,
                            # saturation=0.4),
            transforms.ToTensor(),
            # Lighting(0.1),
            normalize])
    else:
        transform_train = transform_test
    dataset_train = datasets.ImageFolder(traindir, transform_train)
    dataset_val = datasets.ImageFolder(valdir, transform_test)
    dataset_test = datasets.ImageFolder(valdir, transform_test)
    #dataset_test = datasets.ImageFolder(testdir, transform_test)
    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=8, split=False)

def loaders_tiny_imagenet(dataset, batch_size, cuda, n_classes,
                     train_size=100000, augment=True, val_size=10000,
                     test_size=10000, test_batch_size=128, topk=None, noise=False,
                     multiple_crops=False, data_root=None, **kwargs):
    assert dataset == 'tiny_imagenet'
    data_root = data_root if data_root is not None else os.environ['VISION_DATA']
    root = '{}/tiny-imagenet-200'.format(data_root)
    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2302, 0.2265, 0.2262]
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    testdir = os.path.join(root, 'test')
    normalize = transforms.Normalize(mean=mean, std=std)
    if multiple_crops:
        print('Using multiple crops')
        transform_test = transforms.Compose([
            transforms.Pad((4,4)),
            transforms.TenCrop(64),
            lambda x: [normalize(transforms.functional.to_tensor(img)) for img in x]])
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    if augment:
        transform_train = transforms.Compose([
            transforms.Pad((4,4)),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transform_test
    dataset_train = datasets.ImageFolder(traindir, transform_train)
    dataset_val = datasets.ImageFolder(valdir, transform_test)
    dataset_test = datasets.ImageFolder(testdir, transform_test)
    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size, test_batch_size, cuda, n_classes, num_workers=4, split=False)

def loaders_matrix_fac(dataset, batch_size=100, cuda=0,
                  train_size=800, val_size=200, test_size=200,
                  test_batch_size=200, augment=False, **kwargs):

    assert 'matrix_fac' in dataset
    root = '{}/{}'.format(os.environ['VISION_DATA'], "matric_fac")
    fname_save = os.path.join(root,dataset)
    fname = fname_save
    val_size = int(train_size * 0.25)
    test_size = val_size

    if not os.path.isfile(fname):
        data = generate_synthetic_matrix_factorization_data(nsamples=train_size+val_size)
        # print('train', train_size, 'val', val_size)
        file = open(fname, 'wb')
        pickle.dump(data, file)
        file.close()

    file = open(fname, 'rb')
    A, y  = pickle.load(file)
    file.close()
    X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=9513451)
    dataset_train = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
    dataset_val = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))
    dataset_test = dataset_val

    train_size = len(dataset_train)
    val_size = len(dataset_val)
    test_size = len(dataset_test)
    test_batch_size = val_size

    return create_loaders(dataset_train, dataset_val,
                          dataset_test, train_size, val_size, test_size,
                          batch_size=batch_size,
                          test_batch_size=test_batch_size,
                          cuda=cuda, n_classes=2, num_workers=0, split=False)

def generate_synthetic_matrix_factorization_data(xdim=6, ydim=10, nsamples=1000, A_condition_number=1e-10):
    """
    Generate a synthetic matrix factorization dataset as suggested by Ben Recht.
        See: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
    """
    # print('creating dataset, nsamples:', nsamples)
    Atrue = np.linspace(1, A_condition_number, ydim).reshape(-1, 1) * np.random.rand(ydim, xdim)
    # the inputs
    X = np.random.randn(xdim, nsamples)
    # the y's to fit
    Ytrue = Atrue.dot(X)
    data = (X.T, Ytrue.T)
    return data

