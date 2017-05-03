
import os
import wget
import os.path as osp
# import scipy.io as sio
import torch.utils.data as data
from spyder.utils import iofuncs


class TextureLoader(data.Dataset):
    url = 'http://157.253.63.7/texturesPublic'
    filename = 'texturesPublic'

    def __init__(self, root, transform=None, train=False,
                 val=False, download=False):
        self.root = root
        self.transform = transform

        if download:
            wget.download(self.url)

        if not osp.exists(self.filename):
            raise RuntimeError('Dataset not found' +
                               ' You can use download=True to download it')

        
