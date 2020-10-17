import json
import xmltodict
import os
import logging
from pathlib import Path
from json import JSONEncoder

dataDirPath = Path("AI/Data/Annotation").absolute()
outputFile = open("traindata.json", "w+")
settings = None
trainingData = list()


class SettingEncoder(JSONEncoder):
    def default(self, o):
        t = []
        keys = o.__dict__.keys()
        for k in keys:
            t[k] = JSONEncoder[k].toJson()
        return o.__dict__

encoder = SettingEncoder

class Setting:
    def __init__(self, folder, filename, size, name):
        self.folder = folder
        self.filename = filename
        self.size = size
        self.name = name

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class Settings():
    def __init__(self, settings=dict()):
        self.settings = settings

    def setSettings(self, settings):
        self.settings = settings

    def findConfigurationByImage(self, imgName):
        for s in self.settings:
            if s.filename == imgName:
                return Setting(s.folder, s.filename, s.size, s.object[0].name)

        return False

    def findConfigurationByDir(self, dir):
        arr = []
        for s in self.settings:
            if dir[1:len(dir)] in s['folder']:
                arr.append(Setting(s['folder'], s['filename'], s['size'], "Test"))
        return arr

    def toJson(self):
        print(lambda o: o.__dict__)
        return json.dumps(self, default=lambda o: o.__dict__)

for d in os.scandir(dataDirPath):
    if d.name.endswith('json'):
        with open(d) as f:
            settings = Settings(json.load(f)['annotation'])

imgDataMap = {}
for d in os.scandir(dataDirPath):
    if not d.name.endswith('json') and not d.name.endswith('xml'):
        dogDirectory = d.name.split('-')[0]
        dataMap = settings.findConfigurationByDir(dogDirectory)
        if d.name not in dataMap:
            imgDataMap[d.name] = []

        imgDataMap[d.name].append(dataMap)

'''
Need to resolve generating the output of 
this dict() so I can feed the train data to main.py
@TODO
'''
def generate():
    print(outputFile)
    outputFile.truncate()
    if len(imgDataMap) > 0:
        outputFile.write(json.dumps(encoder.default(imgDataMap)))
        print("Output file updated!")
    else:
        print("No update was made to the output file")



