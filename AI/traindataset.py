from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import cv2
import torchvision.transforms as transforms
from PIL import Image
import PIL
import numpy as np
import json
import logging
import torch

# asctime: time of the log was printed out
# levelname: name of the log
# datefmt: format the time of the log
# give DEBUG log
rfh = logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.DEBUG,
    filename='logs/logs.trainingdataset.log')

logger = logging.getLogger('my_app')
class DogTrainerSet(Dataset):
    def __init__(self, main_dir, labels):
        self.main_dir = main_dir
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

        parsedLabels = []
        for l in labels:
            parsedLabels.append(l)
        sentenceIdx = np.linspace(0,len(parsedLabels), len(parsedLabels), False)
        torch_idx = torch.tensor(sentenceIdx)
        self.labels =  parsedLabels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        #IDFK somethings wrong here I guess fuck me.
        img_loc = Path(os.path.join(self.main_dir, self.total_imgs[idx]))
        try: 
            image = Image.open(img_loc)
        except Exception as e:
            logger.error(e)
            return False
        image = transforms.ToTensor()(image)
        image = transforms.ToPILImage(mode='RGB')(image)

        image_2_npArray = np.asarray(image)
#        print(np.shape(image_2_npArray))
#        print('the shape of loaded image transformed into numpy array: {}'.format(np.shape(image_2_npArray)))
#        print('transformed image: {}'.format(image_2_npArray))

        # transform the numpy array into the tensor
        sample = transforms.ToTensor()(image_2_npArray)
#        print('the shape of numpy array transformed into tensor: {}'.format(np.shape(image_2_npArray_2_tensor)))
#        print('transformed numpy array: {}'.format(image_2_npArray_2_tensor))
        #You now have all the labels and they are pretty easy to mess with so.
        #Lets return the correct sample and target :D
        id = 0
        return sample, self.labels
        for k in self.labels:
            for f in self.labels[k]:
                if self.total_imgs[idx] == f:
                    target = self.labels[k][f]
                    return sample, 
        

        

    def getImageRaw(self, idx):
        img_loc = Path(os.path.join(self.main_dir, self.total_imgs[idx]))
        image = Image.open(img_loc)
        image = transforms.ToTensor()(image)
        image = transforms.ToPILImage(mode='RGB')(image)

        return image
