import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from utils.dataset import UAVDatasetPatches

def get_test_loader(test_img_dir, test_msk_dir, mean, std, batch_size, num_workers=4, pin_memory=True):

    test_transform = A.Compose(
        [
            A.Normalize(
                mean = mean,
                std = std,
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )
    test_ds = UAVDatasetPatches(img_list=test_img_dir, msk_list=test_msk_dir, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)
    return test_loader

def predict(model, test_loader, device):
    '''
    predicts all images in the test_loader
    '''
    model.eval()
    predictions_whole = None 

    for inputs, targets in test_loader:
        with torch.no_grad():
            predictions = predict_one_batch(model, inputs, targets, device)
            if predictions_whole is None:
                predictions_whole = predictions
            else:
                predictions_whole = torch.cat((predictions_whole, predictions), dim=0)
    return predictions_whole

def predict_one_batch(model, inputs, targets, device):
    '''
    validates one batch
    '''
    with torch.cuda.amp.autocast():
        inputs = inputs.float().to(device=device)
        targets = targets.long().to(device=device)

        predictions = model(inputs)
        probabilities = torch.sigmoid(predictions.squeeze(1))
        predicted_masks = (probabilities >= 0.5).float() * 1
        predicted_masks = torch.argmax(predicted_masks.int(), dim=1)
    return predicted_masks

def convert_labelmap_to_color(labelmap, labels = [(199, 199, 199), (31, 119, 180), (255, 127, 14)]):
    '''
    Colors the 1 channel output into a RGB Image
    '''   
    lookup_table = np.array(labels)
    result = np.zeros((*labelmap.shape,3), dtype=np.uint8)
    np.take(lookup_table, labelmap, axis=0, out=result)
    return result

def combine_labelmap_from_slices(labelmap, grid = (22,15)):
    '''
    input: torch tensor in gpu with shape NxWxH or NxCxWxH
    takes a labelmap of the shape of BxWxH and converts it to WxH, corresponding a whole capture
    '''
    slc_size =256
    if len(labelmap.shape) == 3:
        labelmap = labelmap.cpu().numpy()
        full_ann = np.zeros((grid[1]*slc_size, grid[0]*slc_size),dtype=np.uint8)
        offset = (slc_size,slc_size)
        tile_size= (slc_size,slc_size)
        placement=0
        for i in range(grid[1]):
            for j in range(grid[0]):
                full_ann[offset[1]*i:min(offset[1]*i+tile_size[1], full_ann.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], full_ann.shape[1])] = labelmap[placement]
                placement+=1
    elif len(labelmap.shape) ==4:
        # reshape 
        labelmap = labelmap.permute(0,2,3,1).cpu().numpy()
        
        full_ann = np.zeros((grid[1]*slc_size, grid[0]*slc_size, 3))
        offset = (slc_size,slc_size)
        tile_size= (slc_size,slc_size)
        placement=0
        for i in range(grid[1]):
            for j in range(grid[0]):
                full_ann[offset[1]*i:min(offset[1]*i+tile_size[1], full_ann.shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], full_ann.shape[1]),:] = labelmap[placement]
                placement+=1
    return full_ann

def get_slices_per_image(labelmap, slc_per_image=330):
    '''
    returns a list with the length of images, with the labelmap_per_image as BxWxH as each item
    '''
    num_images = int(labelmap.shape[0]/slc_per_image)
    labelmaps =[]
    for i in range(num_images):
        labelmap_per_image = labelmap[i*slc_per_image:(i+1)*slc_per_image,:,:] 
        labelmaps.append(labelmap_per_image)
    return labelmaps

def reshape_predictions_to_images(preds, labels, mask_shape =(3648, 5472)):
    predictions_color = []
    if mask_shape == (3648, 5472):
        slc_per_image = 330
        grid = (22,15)
    elif mask_shape == (2816, 2560):
        slc_per_image =110
        grid = (10,11)
    else: 
        raise NotImplementedError(f"{mask_shape=} not implemented.")

    preds_labelmaps = get_slices_per_image(preds, slc_per_image=slc_per_image)
    for lab in preds_labelmaps:
        lab_full = combine_labelmap_from_slices(lab, grid=grid)
        lab_full = lab_full[0:mask_shape[0], 0:mask_shape[1]]
        prediction = convert_labelmap_to_color(lab_full, labels=labels)
        predictions_color.append(prediction)       
    return predictions_color
