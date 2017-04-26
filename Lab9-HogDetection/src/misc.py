import cv2
import numpy as np


# def get_max_size_bounding_box(bbx):
#     max_bbx, img_max = np.array([0, 0]), None
#     for key in bbx:
#         print(key)
#         bbx_img = bbx[key]
#         if len(bbx_img.shape) == 1:
#             max_img_bbx = bbx_img[2:]
#         else:
#             max_img_bbx = np.amax(bbx_img[:, 2:], axis=0)
#         if np.any(max_img_bbx > max_bbx):
#             max_bbx, img_max = max_img_bbx, key
#     return max_bbx, img_max


# def get_mean_size_bounding_box(bbx):
#     mean_bbx, count = np.array([0, 0]), 0
#     for key in bbx:
#         print(key)
#         bbx_img = bbx[key]
#         if len(bbx_img.shape) == 1:
#             bbx_img = bbx_img.reshape(1, 4)
#         bbx_img_sum = np.sum(bbx_img[:, 2:], axis=0)
#         mean_bbx += bbx_img_sum
#         count += bbx_img.shape[0]
#     mean_bbx = mean_bbx / count
#     return np.ceil(mean_bbx)


# def get_dataset_bounding_boxes(bbx, path, dim):
#     pos = []
#     count = 0
#     dim_xy = dim / HOG_SIZE_CELL
#     hog_dim = (int(dim_xy[0]), int(dim_xy[1]), 31)
#     print(hog_dim)
#     mean_template = np.zeros(hog_dim)
#     for dirpath, dirs, files in os.walk(path):
#         bar = progressbar.ProgressBar(redirect_stdout=True)
#         for file in bar(files):
#             basename, _ = osp.splitext(file)
#             img_path = osp.join(dirpath, file)
#             print(img_path)
#             img_bbx = bbx[basename]
#             if len(img_bbx.shape) == 1:
#                 img_bbx = img_bbx.reshape(1, len(img_bbx))
#             img = mpimg.imread(img_path)
#             # print(img_bbx.shape)
#             for i in range(0, img_bbx.shape[0]):
#                 x, y, w, h = img_bbx[i, :]
#                 # print(img.shape, (x, y, w, h))
#                 img_cropped = img[y:y + h, x: x + w]
#                 try:
#                     res = cv2.resize(img_cropped, tuple(np.int64(dim)),
#                                      interpolation=cv2.INTER_CUBIC)
#                 except Exception:
#                     continue
#                 res = np.transpose(res, [1, 0, 2])
#                 # res = imresize(img_cropped, tuple(np.int64(dim)))
#                 # print(res.shape)
#                 # hog_feat = hog(res, HOG_SIZE_CELL)
#                 hog_feat = hog_features(res)
#                 # print(hog_feat.shape)
#                 mean_template += hog_feat
#                 pos.append(hog_feat)
#                 count += 1
#     return pos, mean_template / count


def hog_features(img):
    cell_size = (8, 8)  # h x w in pixels
    block_size = (2, 2)  # h x w in cells
    nbins = 31  # number of orientation bins
    win_size = (img.shape[1] // cell_size[1] * cell_size[1],
                img.shape[0] // cell_size[0] * cell_size[0])

    # winSize is the size of the image cropped to an multiple of the cell size
    hog = cv2.HOGDescriptor(_winSize=win_size,
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
    hog_feats = hog.compute(img) \
        .reshape(n_cells[1] - block_size[1] + 1,
                 n_cells[0] - block_size[0] + 1,
                 block_size[0], block_size[1], nbins) \
        .transpose((1, 0, 2, 3, 4))  # index blocks by rows first
    # hog_feats now contains the gradient amplitudes for each direction,
    # for each cell of its group for each group.
    # Indexing is by rows then columns.

    gradients = np.zeros((n_cells[0], n_cells[1], nbins))

    # count cells (border cells appear less often across overlapping groups)
    cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

    for off_y in range(block_size[0]):
        for off_x in range(block_size[1]):
            gradients[off_y:n_cells[0] - block_size[0] + off_y + 1,
                      off_x:n_cells[1] - block_size[1] + off_x + 1] += \
                hog_feats[:, :, off_y, off_x, :]
            cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                       off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

    # Average gradients
    gradients /= cell_count
    return gradients


def collect_uniform_integers(a, b, N):
    step = np.floor((b - a) / (N - 1))
    return np.arange(0, N) * step + a


def ind2sub(array_shape, ind):
    rows = np.floor(ind.astype('int') / array_shape[1])
    cols = np.floor(ind.astype('int') % array_shape[1])
    return (rows, cols)
