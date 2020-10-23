from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json

'''
Def some issues in the data cleanup and process scripts. Need to fix this
before we can train the AI fully but hey shes ready to learn at least!
Welcome Canis!
'''
class DogTrainerSet(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        all_imgs = os.listdir(main_dir)
        self.dog_classes = json.load(open("traindata.json", 'r'))
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = Path(os.path.join(self.main_dir, self.total_imgs[idx]))
        label = None
        for dogCategory in self.dog_classes.keys():
            for dogRef in self.dog_classes[dogCategory]:
                for ref in dogRef:
                    ref = json.loads(ref)
                    filename = ref['filename'] + ".jpg"
                    if filename == self.total_imgs[idx]:
                        label = ref['name']
        try:
            image = Image.open(img_loc)
        except PIL.UnidentifiedImageError as e:
            return False
        image = transforms.ToTensor()(image)
        image = transforms.ToPILImage(mode='RGB')(image)

        image_2_npArray = np.asarray(image)
#        print(np.shape(image_2_npArray))
#        print('the shape of loaded image transformed into numpy array: {}'.format(np.shape(image_2_npArray)))
#        print('transformed image: {}'.format(image_2_npArray))

        # transform the numpy array into the tensor
        image_2_npArray_2_tensor = transforms.ToTensor()(image_2_npArray)
#        print('the shape of numpy array transformed into tensor: {}'.format(np.shape(image_2_npArray_2_tensor)))
#        print('transformed numpy array: {}'.format(image_2_npArray_2_tensor))
        if label != None:
            sample = image_2_npArray_2_tensor
            target = label
            if target is not None:
                #Idk I need you guys to be tensors?
                target = transform.to_tensor(target)
        else: 
            os.remove(img_loc)
            print("Could not find label")
            return 0, 0
        return sample, target

        

    def getImageRaw(self, idx):
        img_loc = Path(os.path.join(self.main_dir, self.total_imgs[idx]))
        image = Image.open(img_loc)
        image = transforms.ToTensor()(image)
        image = transforms.ToPILImage(mode='RGB')(image)

        return image
