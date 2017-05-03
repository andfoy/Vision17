
import os
import wget
import torch
import errno
import os.path as osp
# import scipy.io as sio
import torch.utils.data as data
from spyder.utils import iofuncs


class TextureLoader(data.Dataset):
    url = 'http://157.253.63.7/texturesPublic'
    filename = 'texturesPublic'
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    val_file = 'val.pt'
    META = 'meta'
    TEXTURES = 'texturesPublic'

    def __init__(self, root, transform=None, train=False,
                 test=False, download=False):
        self.root = root
        self.transform = transform
        self.train = train
        self.test = test

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found' +
                               ' You can use download=True to download it')

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
        wget.download(self.url)

        print("Processing...")
        train_set, test_set, val_set = self.read_dataset()

        train_path = os.path.join(self.root, self.processed_folder,
                                  self.training_file)
        test_path = os.path.join(self.root, self.processed_folder,
                                 self.test_file)
        val_path = os.path.join(self.root, self.processed_folder,
                                self.val_file)

        with open(train_path, 'wb') as fp:
            torch.save(train_set, fp)

        with open(test_path, 'wb') as fp:
            torch.save(test_set, fp)

        with open(val_path, 'wb') as fp:
            torch.save(val_set, fp)

        print('Done!')

    def read_dataset(self):
        data, _ = iofuncs.load_matlab(self.filename)
        meta_info = data[self.META]
        dataset = data[self.TEXTURES]

        classes = meta_info['classes']
        class_to_idx = {classes[i]: i + 1 for i in range(0, len(classes))}

        sets = meta_info['sets']
        # set_to_idx = {sets[i]: i + 1 for i in range(0, len(sets))}

        data = {}
        textures = dataset['data']
        for i, img_set in sets:
