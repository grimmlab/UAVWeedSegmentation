from pathlib import Path
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
from utils.parser import create_compare_parser
from utils.compare import (
    load_img_gt_pd,
    print_metrics,
    plot_confusion_matrix,
)


args = create_compare_parser()

subset = args.subset
bbch = args.subset
save_path = Path(f"results/predictions/")

if subset == "test":
    img, gt, pred = load_img_gt_pd(subset=subset)
    print(f"Calculating metrics on subset {subset}...")
    cm = print_metrics(gt, pred)
    print(f"Plotting Figure 3 to {save_path}...")
    fig = plot_confusion_matrix(cm*100, figsize=(15,12))
    fig.savefig(f'{save_path}/cm_subset_{subset}.pdf', bbox_inches='tight', dpi=300)
elif subset == "test_different_bbch":
    bbch = args.bbch
    if bbch == None:
        raise ValueError("bbch cannot be None for subset test_different_bbch")
    if bbch == "15" or bbch == "19":
        img, gt, pred = load_img_gt_pd(subset=subset, bbch=bbch)
        print(f"Calculate metrics on subset {subset}, with bbch {bbch}...")
        cm = print_metrics(gt, pred)
        print(f"Plotting part of Figure 5 to {save_path}...")
        fig = plot_confusion_matrix(cm*100, figsize=(15,12))
        fig.savefig(f'{save_path}/cm_subset_{subset}_bbch_{bbch}.png', bbox_inches='tight', dpi=300)
    else:
        raise NotImplementedError(f"Only BBCH 15 and 19 in current test set, not {bbch}")