import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import cv2
from torch.utils.data import Dataset
import numpy as np
import torch


def split_data(data_path):
        
    text_ann_path = os.path.join(data_path, "text_annotations.xlsx")

    # Read excel file and split names
    text_ann = pd.read_excel(text_ann_path)
    image_names = text_ann['Image'].tolist()
    train_images, temp_images = train_test_split(image_names, train_size=2183, random_state=50)
    val_images, test_images = train_test_split(temp_images, test_size=273, random_state=50)

    # Create folders and Copy files to each folder
    copy_files(text_ann,train_images, data_path, 'train')
    copy_files(text_ann,val_images, data_path, 'val')
    copy_files(text_ann, test_images, data_path, 'test')

 
    print(f"Data split completed: {len(train_images)} training, {len(val_images)} validation, {len(test_images)} testing.")
    


def copy_files(text_ann, image_set, data_path, folder):

    new_img_path = os.path.join(data_path, folder,'frames')
    new_mask_path = os.path.join(data_path, folder, 'masks')

    os.makedirs(new_img_path,exist_ok=True)
    os.makedirs(new_mask_path,exist_ok=True)

    for img in image_set:
        img_path = os.path.join(data_path, 'frames', img)
        mask_path = os.path.join(data_path, 'masks', img)
            
        shutil.copy(img_path, os.path.join(new_img_path, img))
        shutil.copy(mask_path, os.path.join(new_mask_path, img))

    text_annotations = text_ann[text_ann['Image'].isin(image_set)]
    text_annotations.to_excel(os.path.join(data_path, folder, 'text_annotations.xlsx'), index=False)




# Resize, Augment and create Pytorch tensors for dataset

class SegmentationDataset(Dataset):
    def __init__(self, data_path, folder, transform=None, target_size=(224, 224)):

        self.img_folder = os.path.join(data_path, folder, "frames")
        self.mask_folder = os.path.join(data_path, folder, "masks")
        self.image_names = os.listdir(self.img_folder)
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_folder, img_name)
        mask_path = os.path.join(self.mask_folder, img_name)

        # Load image and mask
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        #print('unique values in masks',np.unique(mask))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        #print(f"Mask shape after transform: {mask.shape}, dtype: {mask.dtype}") 

        
        mask = np.where(mask==255,1,0).astype(np.uint8) # Ensure mask is binary
        image = image.to(torch.float32) / 255.0 # Convert image to float32 and normalize
        mask = torch.from_numpy(mask).unsqueeze(0).to(torch.long)  # Add channel dimension and convert to long

        #print(f"Mask shape after conversion: {mask.shape}, dtype: {mask.dtype}, type: {type(mask)}")

        
        return image, mask