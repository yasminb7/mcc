import numpy as np
import os
import matplotlib.pyplot as plt

def convertImage(path):
    """Load an image (for a maze) from a given path and do some transformations on it."""
    data = plt.imread(path)
    data = np.flipud(data)
    data = np.swapaxes(data, 0, 1)
    assert np.amax(data)<=1.0, "Array maximum should be 1.0"
    assert np.amin(data)>=0.0, "Array minimum should be >=0.0"
    data = np.ones_like(data) - data
    np.save(path + ".npy", data)
    return data

if __name__ == "__main__":
    cwd = os.getcwd()
    folder = "resources/" 
    #maze = "easy_1600.png"
    maze = "hard_1600.png"
    #maze = "eat_15.png"  
    #maze = "maze_empty_1600.png"
    convertImage(os.path.join(cwd, folder, maze))