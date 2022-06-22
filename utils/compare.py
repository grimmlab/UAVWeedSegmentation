from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from kornia.utils.one_hot import one_hot
from utils.patch_utils import load_image, load_mask
from sklearn.metrics import confusion_matrix, classification_report
import torch

def get_img_gt_pd_ls(subset, bbch=None):
    gt_path=Path(f"./data/{subset}/")
    pd_path=Path("./results/predictions/")
    if bbch:
        bbch_str = f"bbch{bbch}"
        gt_ls = list(gt_path.glob(f'msk/*{bbch_str}*.png'))
        gt_ls = sorted(gt_ls)
        pd_ls = list(pd_path.glob(f'{subset}/*{bbch_str}*.png'))
        pd_ls = sorted(pd_ls)
        img_ls = list(gt_path.glob(f'img/*{bbch_str}*.jpg'))
        img_ls = sorted(img_ls)
    else:
        gt_ls = list(gt_path.glob(f'msk/*.png'))
        gt_ls = sorted(gt_ls)
        pd_ls = list(pd_path.glob(f'{subset}/*.png'))
        pd_ls = sorted(pd_ls)
        img_ls = list(gt_path.glob(f'img/*.jpg'))
        img_ls = sorted(img_ls)
    return img_ls, gt_ls, pd_ls

def get_img_gt_pd_arrs(img_ls, gt_ls, pd_ls):
    '''
    takes lists of gt paths and pd paths and returns arrays of shape NxWxH, where N is the number of images
    '''
    gts = []
    pds = []
    imgs = []
    for img, gt, pred in zip(img_ls, gt_ls, pd_ls):
        img_arr = load_image(img)
        gt_arr = load_mask(gt)
        pd_arr = load_mask(pred)
        gts.append(gt_arr)
        imgs.append(img_arr)
        pds.append(pd_arr)
    
    img = np.array(imgs)
    gt = np.array(gts)
    pred = np.array(pds)
    return img, gt, pred

def load_img_gt_pd(subset, bbch=None):
    img_ls, gt_ls, pd_ls= get_img_gt_pd_ls(subset=subset, bbch=bbch)
    img, gt, pred = get_img_gt_pd_arrs(
        img_ls=img_ls, 
        gt_ls=gt_ls, 
        pd_ls=pd_ls)
    return img, gt, pred

def print_metrics(gt, pred):
    cr_df, cm = calc_pixel_metrics(
        gt=gt, 
        pd=pred, 
        normalize="true")
        
    print("Classification Report:")
    print(cr_df)
    print("Confusion Matrix")
    print(cm)
    _, dice_score = dice_loss_hard(pred=pred, gt=gt, num_classes=3)
    print(f"Dice Score: {dice_score}")
    return cm

def calc_pixel_metrics(gt, pd, target_names =["Background", "Sorghum", "Weed"], labels=[0,1,2], normalize=None):
    '''
    takes list of numpy arrays of anns and their corresponding predictions 
    calculates the confusion matrix 
    returns numpy 2D array
    '''
    ann_lbl = gt.reshape(-1)
    pred_lbl = pd.reshape(-1)
    cm = confusion_matrix(ann_lbl, pred_lbl, labels=labels, normalize=normalize)
    cr = classification_report(ann_lbl, pred_lbl, target_names=target_names, labels=labels, output_dict=True)
    cr_df = classification_report_to_df(cr)
    return cr_df, cm

def dice_loss_hard(pred: np.array, gt: np.array, num_classes:int ,eps: float = 1e-8):
    r"""
    NOTE: 
    Adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/dice.html#dice_loss
    instead of softmax, it uses the one-hot-encoding of the Ground Truth and
    to return the mean of dice_scores also
    Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        pred: Prediction numpy array with shape :math:`(N, H, W)` 
        gt: Ground Truth labels numpy array with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    """
    pred = torch.tensor(pred, dtype=torch.int64).cpu()
    gt = torch.tensor(gt, dtype=torch.int64).cpu()

    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"pred type is not a torch.Tensor. Got {type(pred)}")

    if not len(pred.shape) == 3:
        raise ValueError(f"Invalid pred shape, we expect BxHxW. Got: {pred.shape}")

    if not pred.shape[-2:] == gt.shape[-2:]:
        raise ValueError(f"pred and gt shapes must be the same. Got: {pred.shape} and {gt.shape}")

    if not pred.device == gt.device:
        raise ValueError(f"pred and gt must be in the same device. Got: {pred.device} and {gt.device}")

    # compute softmax over the classes axis
    input_one_hot: torch.Tensor = one_hot(pred, num_classes=num_classes, device=pred.device)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(gt, num_classes=num_classes, device=gt.device)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_one_hot * target_one_hot, dims)
    cardinality = torch.sum(input_one_hot + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)

    return torch.mean(-dice_score + 1.0), torch.mean(dice_score)

def classification_report_to_df(cr):
    """
    input: classification report dict from sklearn
    
    output: accuracy, pandas df without accuracy in it
    """
    acc = cr["accuracy"]
    r = dict(cr)
    del r["accuracy"]
    df = pd.DataFrame(r).T
    return acc, df


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def plot_confusion_matrix(cm, display_labels=["BG","S","W"] ,xticks_rotation="horizontal", values_format=".1f", figsize=(8,8)):
    '''
    adapted from https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/metrics/_plot/confusion_matrix.py#L12
    '''

    if display_labels is None:
        display_labels = np.arange(cm.shape[0])
    fig, ax = plt.subplots(figsize=cm2inch(figsize))
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap = 'Blues')
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    text_ = np.empty_like(cm, dtype=object)
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm[i, j] < thresh else cmap_min


        text_cm = format(cm[i, j], values_format)
        if text_cm == f"{0:.1f}":
            text_cm = "<0.1"
        elif text_cm == f"{100:.1f}":
            text_cm = ">99.9"

        text_[i, j] = ax.text(
            j, i, text_cm,
            ha="center", va="center",
            color=color)
    im_.set_clim(0,100)
    fig.colorbar(im_, ax=ax)

    ax.set(xticks=np.arange(n_classes),
                   yticks=np.arange(n_classes),
                   xticklabels=display_labels,
                   yticklabels=display_labels,
                   ylabel="True label",
                   xlabel="Predicted label")
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation);
    return fig
