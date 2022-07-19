import cv2
from torch.utils.data import Dataset

class UAVDatasetPatches(Dataset):
    def __init__(self, img_list, msk_list, transform=None):
        '''
        img_ls: list of image Paths to load
        msk_ls: list of mask Paths to load
        loads the dataset from a list of images and masks
        '''
        self.transform = transform
        self.img_list= img_list
        self.msk_list= msk_list
        assert len(self.img_list) == len(self.msk_list), "Image and Mask Patches have different lengths."
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.img_list[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.msk_list[idx]), cv2.IMREAD_GRAYSCALE) # load as np.float32
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask
