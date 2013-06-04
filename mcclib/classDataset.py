import numpy as np
import os
import constants
import utils
from utils import info
import cPickle

ending = ".npy"

class ArrayFile(object):
    """Handles information regarding arrays and how to store them on disk."""
    def __init__(self, name, dim, dtype, filename=None, fileprefix="A", onDisk=False):
        self.name = name
        self.dim = dim
        self.dtype = dtype
        self.fileprefix = fileprefix
        self.onDisk = onDisk
    
    def getFilename(self):
        """Returns the filename for the particular array."""
        if self.fileprefix is not None:
            filename = self.fileprefix + "_" + self.name + ending
        else:
            filename = self.name + ending
        return filename

class Dataset(object):
    """Provides help in creating, saving, and reloading (if necessary) the dataset belonging to a simulation."""
    is_amoeboid = constants.TYPE_AMOEBOID
    is_mesenchymal = constants.TYPE_MESENCHYMAL
    
    AMOEBOID = {"times" : ArrayFile("times", "NNN", constants.floattype),
                "types" : ArrayFile("types", "N_agents", np.int_),
              "positions" : ArrayFile("positions", "shapexDim", constants.floattype, onDisk=True),
              "velocities" : ArrayFile("velocities", "shapexDim", constants.floattype, onDisk=True),
              "energies" : ArrayFile("energies", "shapex1", constants.floattype),
              "states" : ArrayFile("states", "shapex1", np.character),
              "statechanges" : ArrayFile("statechanges", "N_agents", constants.floattype),
              "periods" : ArrayFile("periods", "N_agents", constants.floattype),
              "delays" : ArrayFile("delays", "N_agents", constants.floattype),
              "eating" : ArrayFile("eating", "shapex1", np.bool_),
              "direction_angles" : ArrayFile("direction_angles", "N_agents", constants.floattype)}

    def __init__(self, arrays, max_time, dt, N_agents, N_dim, path, doAllocate=True, fileprefix=None):
        """Constructs a new dataset for the given parameters.
        
        If `doAllocate` is *False* then nothing will be written to disk."""
        self.dt = dt
        self.arrays = arrays
        self.NNN = int(max_time / dt)
        self.shapexDim = (self.NNN, N_agents, N_dim)
        self.shapex1 = (self.NNN, N_agents)
        self.shapexEdim = (self.NNN, constants.E_dim, N_agents)
        self.N_agents = N_agents
        self.fileprefix = fileprefix
        self.path = path
        
        if doAllocate:
            for name, val in self.arrays.iteritems():
                if val.onDisk:
                    filename = os.path.join(self.path, val.getFilename())
                    arr = np.memmap(filename, dtype=val.dtype, mode='w+', shape=self.__getattribute__(val.dim))
                    setattr(self, name, arr)
                else:
                    setattr(self, name, np.zeros(self.__getattribute__(val.dim), dtype=val.dtype))
        else:
            for name, val in self.arrays.iteritems():
                setattr(self, name, np.empty(self.__getattribute__(val.dim), dtype=val.dtype))
    
    def getTotalSize(self):
        """Returns the total size of the dataset (in bytes)."""
        return self.getSize(False)+self.getSize(True)
    
    def getSize(self, onDisk):
        """Returns the size of the arrays in memory if `onDisk` is *False* or on disk if `onDisk` is *True*."""
        totalbytes = 0
        for name, val in self.arrays.iteritems():
            if val.onDisk==False:
                a = self.__getattribute__(name)
                totalbytes += a.size * a.itemsize
        return totalbytes
                
    def getHumanReadableSize(self):
        """Returns the total size of the dataset (in megabytes)."""
        size = self.getTotalSize()
        bytesInMB = 1024*1024
        return "%s MB" % int(size/bytesInMB) 
    
    def resizeTo(self, iii):
        """Obsolete. If the simulation stops too early, resize all the arrays to `iii` timesteps."""
        if iii==self.NNN:
            return
        self.NNN = iii
        self.shapexDim = (iii, self.shapexDim[1], self.shapexDim[2])
        self.shapex1 = (iii, self.shapex1[1])
        
        for name, val in self.arrays.iteritems():
            self.__getattribute__(name).resize(self.__getattribute__(val.dim), refcheck=False)
    
    def saveTo(self, path):
        """Save the dataset to `path`."""
        for name, val in self.arrays.iteritems():
            arr = self.__getattribute__(name)
            if type(arr) == np.memmap:
                filename = os.path.join(path, val.getFilename() + ".shape.pickle")
                with open(filename, "w") as picklefile:
                    cPickle.dump(arr.shape, picklefile)
                info("Memmap %s saved" % name)
                del arr
            else:
                filepath = os.path.join(path, val.getFilename())
                np.save(filepath, arr)
    
    def erase(self):
        """Erase the dataset."""
        mmapfiles = []
        for name in self.arrays:
            arr = self.__getattribute__(name)
            if type(arr) == np.memmap:
                mmapfiles.append(arr.filename)
                del arr
        for delfile in mmapfiles:
            utils.deleteFile(delfile)
    
#    def release(self):
#        for name in self.arrays:
#            self.__delattr__(name)

def load(arrays, path, fileprefix, dt, readOnly=True):
    """Loads and returns a *dataset* with the specified *array* configuration from path, using a certain *fileprefix*.
    
    Returns `None` in case of failure."""
    try:
        loaded_arrays = dict()
        for name, val in arrays.iteritems():
            filepath = os.path.join(path, val.getFilename())
            if val.onDisk==False:
                loaded_arrays[name] = np.load(filepath)
            else:
                with open(filepath + ".shape.pickle") as picklefile:
                    arrshape = cPickle.load(picklefile)
                    arrmode = "r" if readOnly else "w+"
                    arr = np.memmap(filepath, dtype=val.dtype, mode=arrmode, shape=arrshape)
                    loaded_arrays[name] = arr
    
        sample_arr = loaded_arrays["energies"]
        
        ds = Dataset(arrays, sample_arr.shape[0], dt, sample_arr.shape[1], constants.DIM, path, doAllocate=False, fileprefix=fileprefix)
        for name, val in arrays.iteritems():
            ds.__setattr__(name, loaded_arrays[name])
    except IOError, ex:
        print ex
        return None
    return ds
