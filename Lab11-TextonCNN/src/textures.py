
import os
import wget
import torch
import errno
import pickle
import numpy as np
import os.path as osp
from PIL import Image
import torch.utils.data as data
from spyder.utils import iofuncs


class TextureLoader(data.Dataset):
    url = 'http://157.253.63.7/texturesPublic'
    filename = 'textureDataset.mat'
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    val_file = 'val.pt'
    class_file = 'classidx.pickle'
    META = 'meta'
    TEXTURES = 'texturesPublic'

    def __init__(self, root, transform=None, target_transform=None,
                 train=False, test=False, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.test = test
        self.imgs = None
        self.labels = None
        self.class_to_idx = None

        train_path = os.path.join(self.root, self.processed_folder,
                                  self.training_file)
        test_path = os.path.join(self.root, self.processed_folder,
                                 self.test_file)
        val_path = os.path.join(self.root, self.processed_folder,
                                self.val_file)
        class_path = os.path.join(self.root, self.processed_folder,
                                  self.class_file)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found' +
                               ' You can use download=True to download it')

        with open(class_path, 'rb') as fp:
            self.class_to_idx = pickle.load(fp)

        if train:
            self.imgs, self.labels, self.idx = torch.load(train_path)
        elif test:
            self.imgs, self.labels, self.idx = torch.load(test_path)
        else:
            self.imgs, self.labels, self.idx = torch.load(val_path)

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, idx):
        img, target, img_idx = (self.imgs[idx, :, :], self.labels[idx],
                                self.idx[idx])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, img_idx

    def _check_exists(self):
        train_path = os.path.join(self.root, self.processed_folder,
                                  self.training_file)
        test_path = os.path.join(self.root, self.processed_folder,
                                 self.test_file)
        val_path = os.path.join(self.root, self.processed_folder,
                                self.val_file)
        exists = osp.exists(train_path) and osp.exists(test_path)
        return exists and osp.exists(val_path)

    def download(self):
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print("Downloading: {0}".format(self.url))
        path = osp.join(self.root, self.raw_folder, self.filename)
        wget.download(self.url, out=path)

        print("\nProcessing...")
        train_set, test_set, val_set, class_to_idx = self.read_dataset()

        train_path = os.path.join(self.root, self.processed_folder,
                                  self.training_file)
        test_path = os.path.join(self.root, self.processed_folder,
                                 self.test_file)
        val_path = os.path.join(self.root, self.processed_folder,
                                self.val_file)
        class_path = os.path.join(self.root, self.processed_folder,
                                  self.class_file)

        with open(train_path, 'wb') as fp:
            torch.save(train_set, fp)

        with open(test_path, 'wb') as fp:
            torch.save(test_set, fp)

        with open(val_path, 'wb') as fp:
            torch.save(val_set, fp)

        with open(class_path, 'wb') as fp:
            pickle.dump(class_to_idx, fp, pickle.HIGHEST_PROTOCOL)

        print('Done!')

    def read_dataset(self):
        path = osp.join(self.root, self.raw_folder, self.filename)
        data, _ = iofuncs.load_matlab(path)
        meta_info = data[self.META]
        dataset = data[self.TEXTURES]

        classes = meta_info['classes']
        class_to_idx = {classes[i]: i + 1 for i in range(0, len(classes))}

        sets = meta_info['sets']
        # set_to_idx = {sets[i]: i + 1 for i in range(0, len(sets))}

        data = {}
        textures = dataset['data']
        textures_idx = dataset['id'].ravel()
        textures_set = dataset['set']
        texture_labels = dataset['label'].ravel()
        for i, img_set in enumerate(sets):
            idx = (textures_set == i + 1)[0]
            imgs = textures[:, :, idx]
            labels = texture_labels[idx] - 1
            img_idx = textures_idx[idx]

            imgs = np.transpose(imgs, (-1, 0, 1))
            labels = torch.ByteTensor(labels)
            imgs = torch.ByteTensor(imgs)
            img_idx = torch.ShortTensor(img_idx)

            data[img_set] = (imgs, labels, img_idx)

        return data['train'], data['test'], data['validation'], class_to_idx
