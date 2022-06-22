from pathlib import Path
import numpy as np
from utils.patch_utils import (
    get_file_lists, load_images, load_masks, save_patches, get_idx_to_remove
)
from utils.parser import generate_patches_parser

args = generate_patches_parser()

root_path:str = args.root_path
subset_list = ["trainval", "test", "test_different_bbch"]
data_path = Path(root_path) / "data" 

for subset in subset_list:
    img_list, msk_list = get_file_lists(
                data_path=data_path,
                subset=subset)


    path_to_save_img = data_path / subset / "patches" / "img"
    path_to_save_img.mkdir(parents=True, exist_ok=True)
    path_to_save_msk = data_path / subset / "patches" / "msk"
    path_to_save_msk.mkdir(parents=True, exist_ok=True)
    print(f"Loading images from {data_path / subset}...")
    imgs = load_images(img_ls=img_list)
    print(f"Loading masks from {data_path / subset}...")
    anns = load_masks(msk_ls=msk_list)
    if subset == "trainval":
        remove_idx = get_idx_to_remove(anns)
        print(f"Removing {len(remove_idx)} patches due to too little plants")
        anns = np.delete(anns, remove_idx, axis=0)
        imgs = np.delete(imgs, remove_idx, axis=0)

    print(f"Saving {imgs.shape[0]} images as patches to {path_to_save_img}")
    save_patches(patches=imgs, path_to_save=path_to_save_img, subset=subset, postfix="img")
    print(f"Saving {anns.shape[0]} masks as patches to {path_to_save_msk}")
    save_patches(patches=anns, path_to_save=path_to_save_msk, subset=subset, postfix="msk")