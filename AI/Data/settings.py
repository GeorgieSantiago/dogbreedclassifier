import Setting

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