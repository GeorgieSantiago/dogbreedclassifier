import sys
from AI.Data.cleaner import generate
from pathlib import Path
import json
import argparse
import numpy as np
from AI.canis import canisAI, criterion, optimizer
from AI.traindataset import DogTrainerSet
from torch.utils.data import DataLoader
from torch import device, cuda, save
import matplotlib.pyplot as plt
from tqdm import tqdm
from debugger import logger

hasCuda = cuda.is_available()
device = device("cuda:0" if hasCuda else "cpu")
if cuda.is_available():
    print("Running with Cuda")
else:
    print('Cuda is not available. Performance will be based on available CPU')

def showImg(img):
    try: 
        #img = np.asarray(img)
        #plt.imshow(img.shape( img.squeeze() ))
#        a = np.asarray(img)
        plt.imshow(img)
        plt.figure()
        plt.show()
        #plt.imshow(np.transpose(parsedImg, (1, 3, 0)))
    except ValueError as e:
        logger.error(e)
        exit(-1)

def run():
    # Use a breakpoint in the code line below to debug your script.
    generate(toResize=False)
    dataSchema = json.load(open(Path('traindata.json'), 'r'))
    trainingDataset = DogTrainerSet(Path("AI/Data/FlatImgStore/resized_128_flat_training_images"))
    trainLoader = DataLoader(trainingDataset, batch_size=10, shuffle=True,
                                   num_workers=4, drop_last=False)
    for epoch in range(2):
        running_loss = 0.0
        i = 0
        for image, label in tqdm(trainLoader):
            #Check the response of data and make sure its an 
            optimizer.zero_grad()
            if i % 100 == 0:
                #showImg(trainingDataset.getImageRaw(i))
                i += 1
            try: 
                outputs = canisAI.forward(image, label)
                loss = criterion(outputs, label)
                loss.backward()
            except AttributeError as e:
                logger.error(e)
            optimizer.step()
            running_loss = 0.0
        print('Finished Training')
        PATH = './canis_ai.pth'
        save(canisAI.state_dict(), PATH)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the dog breed classifier')
    parser.add_argument("--resize", type=bool, default=False,
                        help="Should preprocess images")
    parser.add_argument("--training", type=bool, default=True,
                        help="Configure training")
    run()
