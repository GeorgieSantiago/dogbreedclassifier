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


for d in os.scandir(dataDirPath):
    if d.name.endswith('json'):
        with open(d) as f:
            settings = json.load(f)['annotation']

imgDataMap = {}
for d in os.scandir(dataDirPath):
    if not d.name.endswith('json') and not d.name.endswith('xml'):
        dSplitInfo = d.name.split('-')
        dogDirectory = dSplitInfo[0]
        _dogType = dSplitInfo[1:]
        dogType = ""
        for dt in _dogType:
            dogType += dt
        dataMap = findConfigurationByDir(dogDirectory)
        if d.name not in dataMap:
            imgDataMap[d.name] = []
        if dogType not in labels:
            labels.append(dogType.title())
        imgDataMap[d.name].append(dataMap)

'''
Need to resolve generating the output of 
this dict() so I can feed the train data to main.py
@TODO
'''


def generate(toResize=False):
    outputFile.truncate()
    if len(imgDataMap) > 0:
        jsonContent = json.dumps(imgDataMap, indent=0)
        outputFile.write(jsonContent)
        print("Output file updated!")
    else:
        print("No update was made to the output file")

    if toResize:
        inDir = Path('AI/Data/FlatImgStore/flat_training_images')
        outDir = Path('AI/Data/FlatImgStore/resized_128_flat_training_images')
        size = 128
        print('Removing old images...')
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
            print('Images removed')
        if resize(inDir, outDir, size):
            print("Images resized to 128px. Output: " + outDir.name)