import numpy as np
import os.path

def loadImage(path):
    arr = None
    filename = path + ".npy"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            arr = np.load(f)
    else:
        os.system("python png2npy.py %s" % path)
    return arr

#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

#    def plotmaze(self, maze, filename = "maze.png"):
#        fig = plt.figure()
#        ax = fig.add_subplot(111)
#        ax.imshow(maze.T, extent=self.const["fieldlimits"], cmap=cm.get_cmap("binary"))
#        plt.grid(True)
#        plt.axis(self.const["fieldlimits"])
#        mazefilepath = utils.getResultsFilepath(self.resultsdir, filename)
#        plt.savefig(mazefilepath)
#        
#    def plotcgrad(self, concentrationfield, filename = 'concentrationfield.png'):
#        fig = plt.figure(figsize=(16,16), frameon=False)
#        ax = plt.Axes(fig, [0., 0., 1., 1.], )
#        ax.set_axis_off()
#        fig.add_axes(ax)
#        img = utils.scaleToMax(1.0, -concentrationfield.T)
#        ax.imshow(img, extent=self.const["fieldlimits"], origin='lower')
#        concentrationfilepath = utils.getResultsFilepath(self.resultsdir, filename)
#        plt.savefig(concentrationfilepath, bbox_inches='tight', dpi=100)