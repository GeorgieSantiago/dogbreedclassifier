import sys
from AI.Data.cleaner import generate, flat_image_folder
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
import types


def showImg(img):
    try: 
        #img = np.asarray(img)
        #plt.imshow(img.shape( img.squeeze() ))
#        a = np.asarray(img)
        imgCopy = img
        plt.imshow(imgCopy)
        plt.figure()
        plt.show()
        #plt.imshow(np.transpose(parsedImg, (1, 3, 0)))
    except ValueError as e:
        logger.error(e)

def run(parser=None):
    args = None
    if parser != None:
        args = parser.parse_args()
    else:
        args = types.SimpleNamespace()
        args.resize = False
        args.rebuild = False
        args.training = True
    
    print(cuda.get_arch_list())
    device("cpu:0" if not cuda.is_available() else "cuda:0")
    if cuda.is_available():
        print("Running on Cuda")
        exit(1)
    else:
        print("Running on CPU")
        exit(1)
    if args.rebuild:
        flat_image_folder(Path('AI/Data/images'), Path('AI/Data/FlatImgStore/flat_training_images'))
    # Use a breakpoint in the code line below to debug your script.
    dataSchema = generate(toResize=args.resize)
    trainingDataset = DogTrainerSet(Path("AI/Data/FlatImgStore/resized_128_flat_training_images"), dataSchema)
    trainLoader = DataLoader(trainingDataset, batch_size=10, shuffle=True,
                                   num_workers=0, drop_last=False)
    for epoch in range(2):
        running_loss = 0.0
        i = 0
        for image, label in tqdm(trainLoader):
            #idfk something is wrong here I guess fuck me
            #Start Grandma debugging @TODO pussy
            #Check the response of data and make sure its an 
            optimizer.zero_grad()
            if i % 500 == 0:
                showImg(trainingDataset.getImageRaw(i))
            i += 1
            logger.debug("Training image")
            logger.debug(image)
            try: 
                outputs = canisAI(image)
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
    parser.add_argument("--rebuild", type=bool, default=False, 
                        help="Regenerate flat image folder")
    run(parser)
