import json
import os
import argparse
from pathlib import Path
from AI.Data.resizer import resize
import shutil

dataDirPath = Path("AI/Data/Annotation").absolute()
outputFile = open("traindata.json", "w+")
settings = None
labels = list()
trainingData = list()

def findConfigurationByImage(imgName):
    for s in settings:
        if s.filename == imgName:
            return Setting(s.folder, s.filename, s.size, s.object[0].name)
    return False


def findConfigurationByDir(dir):
    arr = []
    for s in settings:
        if dir[1:len(dir)] in s['folder']:
            name = s['object'][0]['name']
            arr.append(Setting(s['folder'], s['filename'], s['size'], name).toJSON())
    return arr


class Setting:
    def __init__(self, folder, filename, size, name):
        self.folder = folder
        self.filename = filename
        self.size = size
        self.name = name

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)

'''
Need to resolve generating the output of 
this dict() so I can feed the train data to main.py
@TODO
'''

def flat_image_folder(input_dir, output_dir):
    print("Removing old flat folder")
    shutil.rmtree(output_dir)
    print("Creating new flat folder")
    os.mkdir(output_dir)
    for (root,dirs,files) in os.walk(input_dir, topdown=True):
        for d in dirs:
            for (root, dirs, files) in os.walk(str(input_dir) + "/" + d):
                for f in files:
                    inputPath = Path(str(input_dir) + "/" + d + "/" + f)
                    outputPath = Path(str(output_dir) + "/" + f)
                    shutil.copy(inputPath, outputPath)
                    print(inputPath)
                    print("Moved to: ")
                    print(outputPath)
def generate(toResize=False) -> dict :
    outputFile.truncate()
    imgDataMap = dict()
    annotations = Path('AI/Data/images')
    for (root,dirs,files) in os.walk(annotations, topdown=True):
        for folder in dirs:
            labelImagePath = 'AI/Data/images/' + folder
            labelImages =  Path(labelImagePath)
            label = folder.split("-")[1:]
            label = "_".join(label)
            imgDataMap[label] = dict()
            for (root, dirs, files) in os.walk(labelImages, topdown=True):
                for f in files:
                    imgDataMap[label][f] = Setting(labelImagePath, f, 128, f)

    if toResize:
        inDir = Path('AI/Data/FlatImgStore/flat_training_images')
        outDir = Path('AI/Data/FlatImgStore/resized_128_flat_training_images')
        print("Removing old resized")
        shutil.rmtree(outDir)
        size = 128
        if resize(inDir, outDir, size):
            print("Images resized to 128px. Output: " + outDir.name)
    return imgDataMap