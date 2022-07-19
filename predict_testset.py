from argparse import _SubParsersAction
import torch
from pathlib import Path
from skimage import io as skio
from utils.patch_utils import get_file_lists, load_image
from utils.train import seed_all, set_model, get_calculated_means_stds_trainval, get_patch_lists
from utils.predict import get_test_loader, predict, reshape_predictions_to_images
from utils.parser import create_test_parser

args = create_test_parser()

subset:str = args.subset
root_path:str = args.root_path
model_save_path:str = args.model
batch_size:int = args.batch_size
device:str = "cuda" if torch.cuda.is_available() else "cpu"

seed_all(seed=args.seed)
print(f"Using Seed {args.seed}")

data_path = Path(root_path) / "data" 
model_save_stem = model_save_path.split('/')[-1]
architecture = model_save_stem.split('_')[1]
encoder_name = model_save_stem.split('_')[2]
replace_stride_with_dilation = model_save_stem.split('_')[3]
b_bilinear = model_save_stem.split('_')[4]
path_to_save = Path(root_path) / "results" / "predictions" / subset
path_to_save.mkdir(parents=True, exist_ok=True)

test_imgs, test_msks = get_patch_lists(
    data_path=data_path, 
    subset=subset)

test_complete_img_ls, _ = get_file_lists(
    data_path, 
    subset=subset)
    
img_shape = load_image(path = str(test_complete_img_ls[0])).shape
means, stds = get_calculated_means_stds_trainval()       

test_loader = get_test_loader(
    test_img_dir=test_imgs, 
    test_msk_dir=test_msks, 
    mean=means, 
    std=stds, 
    batch_size=batch_size,
)

loaded_model = torch.load(model_save_path)
print(f"Loading: {architecture} {encoder_name} ...")
model = set_model(architecture=architecture, encoder_name=encoder_name, pretrained=False, b_bilinear=b_bilinear, replace_stride_with_dilation=replace_stride_with_dilation, num_classes=3).to(device=device)
model.load_state_dict(loaded_model["state_dict"])

print(f"Predicting...")
preds = predict(
    model=model, 
    test_loader=test_loader, 
    device=device)

print("Combining Slices...")
colored_predictions = reshape_predictions_to_images(preds=preds, labels=[(199, 199, 199), (31, 119, 180), (255, 127, 14)], mask_shape =img_shape[:2])
print(f"Saving Predictions to {path_to_save}...")
for preds_to_save, img_name in zip(colored_predictions, test_complete_img_ls):
    skio.imsave(f"{path_to_save}/{img_name.stem}_pred.png", preds_to_save, check_contrast=False)