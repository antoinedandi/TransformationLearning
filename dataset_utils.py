import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def create_rgb_array(size=None, center=None, radius=10):
    # default for mutable params
    if size is None:
        size = np.array([64, 64])
    if center is None:
        center = (0.5 * size).astype(np.int64)
    
    # black image
    rgb_array = 1. * np.zeros((*size, 3), dtype=np.uint8)
    
    # fill function for specific center and radius
    def in_disc(i, j): 
        modulus = (center[0] - (i + 0.5)) ** 2 + (center[1] - (j + 0.5)) ** 2
        bool_val = modulus <= radius ** 2
        return 1. * bool_val

    # draw ball in R coordinate
    rgb_array[:, :, 0] = np.fromfunction(in_disc, size, dtype=np.float64)
    return rgb_array


def create_rgb_array_lozenge(size=None, center=None, scale=5., rot=0.):
    # default lozenge location
    if size is None:
        size = np.array([64, 64])
    if center is None:
        center = (0.5 * size).astype(np.int64)

    # black image
    rgb_array = 1. * np.zeros((*size, 3), dtype=np.uint8)

    # fill function for specific shape parameters
    def in_lozenge(i, j):
        # use center of pixel according to its position relative to the center of the lozenge
        i_centered = i - center[0] + 0.5 * (i < center[0]) - 0.5 * (i > center[0])
        j_centered = j - center[1] + 0.5 * (j < center[1]) - 0.5 * (j > center[1])
        # apply inverse rotation
        i_inv = i_centered * np.cos(-rot) + j_centered * np.sin(-rot)
        j_inv = - i_centered * np.sin(-rot) + j_centered * np.cos(-rot)
        # apply inverse scaling and compute 1 norm (unit ball is a lozenge)
        norm_abs = (1. / scale) * (np.abs(i_inv) + 0.5 * np.abs(j_inv))
        # check if in unit ball
        bool_val = norm_abs <= 1.
        return 1. * bool_val

    # draw lozenge in R coordinate
    rgb_array[:, :, 0] = gaussian_filter(np.fromfunction(in_lozenge, size, dtype=np.float64), sigma=1., mode='reflect')
    return rgb_array

## Creates lozenge outline 
## used for checking predictions
def create_lozenge_outline(target,size = None):
    # default lozenge location
    if size is None:
        size = np.array([64, 64])
    center = target[0:2]
    scale = target[2]
    rot = target[3]

    # black image
    rgb_array = 1. * np.zeros((*size, 3), dtype=np.uint8)

    # fill function for specific shape parameters
    def in_lozenge(i, j):
        # use center of pixel according to its position relative to the center of the lozenge
        i_centered = i - center[0] + 0.5 * (i < center[0]) - 0.5 * (i > center[0])
        j_centered = j - center[1] + 0.5 * (j < center[1]) - 0.5 * (j > center[1])
        # apply inverse rotation
        i_inv = i_centered * np.cos(-rot) + j_centered * np.sin(-rot)
        j_inv = - i_centered * np.sin(-rot) + j_centered * np.cos(-rot)
        # apply inverse scaling and compute 1 norm (unit ball is a lozenge)
        norm_abs = (1. / scale) * (np.abs(i_inv) + 0.5 * np.abs(j_inv))

        ## Difference here
        # check if edge of unit ball
        bool_val = ((norm_abs <= 1.) & (norm_abs >= .9))
        #bool_val = norm_abs == 1.
        return 1. * bool_val

    # draw lozenge outline in G coordinate
    rgb_array[:, :, 1] = np.fromfunction(in_lozenge, size, dtype=np.float64)
    return rgb_array



def show_center(imgs, centers, centers_pred=None, n_samples=16, show_all=False, fontsize=4, pad=1):
    n_total = len(imgs) if len(imgs.shape) == 4 else 1
    if show_all:
        n_samples = n_total
        choices = np.arange(n_total)
    else:
        n_samples = min(n_total, n_samples)
        choices = np.random.choice(n_total, n_samples)
    n_col = min(n_samples, 4)
    n_row = np.ceil(n_samples / n_col)
    cap = np.array([imgs.shape[1] - 1, imgs.shape[2] - 1])
    plt.figure();
    for i, idx in enumerate(choices):
        plt.subplot(n_row, n_col, i + 1);
        img = np.array(imgs[idx]) if len(imgs.shape) == 4 else imgs
        center = centers[idx] if len(centers.shape) == 2 else centers
        plt.xlabel('center : ' + str(center), fontsize=fontsize)
        sub_x, sub_y = np.minimum(np.floor(center).astype(np.int64), cap)
        sup_x, sup_y = np.minimum(np.ceil(center).astype(np.int64), cap)
        img[sup_x, sup_y, :] = 1.
        img[sup_x, sub_y, :] = 1.
        img[sub_x, sup_y, :] = 1.
        img[sub_x, sub_y, :] = 1.
        if centers_pred is not None:
            center_pred = centers_pred[idx] if len(centers_pred.shape) == 2 else centers_pred
            plt.title('center pred: ' + str(center_pred), fontsize=fontsize, pad=pad)
            sub_xpred, sub_ypred = np.minimum(np.floor(center_pred).astype(np.int64), cap)
            sup_xpred, sup_ypred = np.minimum(np.ceil(center_pred).astype(np.int64), cap)
            img[sup_xpred, sup_ypred, [0, 2]] = [0.0, 1.0]
            img[sup_xpred, sub_ypred, [0, 2]] = [0.0, 1.0]
            img[sub_xpred, sup_ypred, [0, 2]] = [0.0, 1.0]
            img[sub_xpred, sub_ypred, [0, 2]] = [0.0, 1.0]
        plt.axis('off')
        plt.imshow(img)


def show_lozenge(imgs, targets, targets_pred=None, n_samples=16, show_all=False, fontsize=4, pad=1):
    n_total = len(imgs) if len(imgs.shape) == 4 else 1
    if show_all:
        n_samples = n_total
        choices = np.arange(n_total)
    else:
        n_samples = min(n_total, n_samples)
        choices = np.random.choice(n_total, n_samples)
    n_col = min(n_samples, 4)
    n_row = np.ceil(n_samples / n_col)
    cap = np.array([imgs.shape[1] - 1, imgs.shape[2] - 1])
    plt.figure();
    centers = targets[:,0:2] if len(targets.shape) == 2 else targets[0:2]
    scales = targets[:,2] if len(targets.shape) == 2 else targets[2]
    rots = targets[:,3] if len(targets.shape) == 2 else targets[3]
    for i, idx in enumerate(choices):
        plt.subplot(n_row, n_col, i + 1);
        img = np.array(imgs[idx]) if len(imgs.shape) == 4 else imgs
        center = centers[idx] if len(centers.shape) == 2 else centers
        plt.xlabel('center : ' + str(center), fontsize=fontsize)
        sub_x, sub_y = np.minimum(np.floor(center).astype(np.int64), cap)
        sup_x, sup_y = np.minimum(np.ceil(center).astype(np.int64), cap)
        img[sup_x, sup_y, :] = 1.
        img[sup_x, sub_y, :] = 1.
        img[sub_x, sup_y, :] = 1.
        img[sub_x, sub_y, :] = 1.
        if targets_pred is not None:
            
            centers_pred = targets_pred[idx,0:2] if len(targets_pred.shape) == 2 else targets_pred[0:2]
            center_pred = centers_pred[idx] if len(centers_pred.shape) == 2 else centers_pred
            plt.title('center pred: ' + str(center_pred), fontsize=fontsize, pad=pad)
            sub_xpred, sub_ypred = np.minimum(np.floor(center_pred).astype(np.int64), cap)
            sup_xpred, sup_ypred = np.minimum(np.ceil(center_pred).astype(np.int64), cap)
            img[sup_xpred, sup_ypred, [0, 2]] = [0.0, 1.0]
            img[sup_xpred, sub_ypred, [0, 2]] = [0.0, 1.0]
            img[sub_xpred, sup_ypred, [0, 2]] = [0.0, 1.0]
            img[sub_xpred, sub_ypred, [0, 2]] = [0.0, 1.0]
            
            t = targets_pred[idx] if len(targets_pred.shape) == 2 else targets_pred
            #print(sum(create_lozenge_outline(t)))
            img = img + create_lozenge_outline(t)
        plt.axis('off')
        plt.imshow(img)
