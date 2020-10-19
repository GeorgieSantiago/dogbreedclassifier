from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class DogTrainerSet(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = Path(os.path.join(self.main_dir, self.total_imgs[idx]))
        image = Image.open(img_loc)
        image = transforms.ToTensor()(image)
        image = transforms.ToPILImage(mode='RGB')(image)

        image_2_npArray = np.asarray(image)
        print(np.shape(image_2_npArray))
        print('the shape of loaded image transformed into numpy array: {}'.format(np.shape(image_2_npArray)))
        print('transformed image: {}'.format(image_2_npArray))

        # transform the numpy array into the tensor
        image_2_npArray_2_tensor = transforms.ToTensor()(image_2_npArray)
        print('the shape of numpy array transformed into tensor: {}'.format(np.shape(image_2_npArray_2_tensor)))
        print('transformed numpy array: {}'.format(image_2_npArray_2_tensor))

        return image_2_npArray_2_tensor
