import sys, os, traceback
import graphics
from mcclib import utils, constants, classDataset
from mcclib.utils import getSimfiles, verifySimulationNecessary
#from utils import getSimfiles, verifySimulationNecessary
from mcclib.classSimulation import Simulation
#import mcclib.classDataset as classDataset
from mcclib.classDataset import Dataset
#import constants
from mcclib.utils import info, error
#from utils import info, error

#probably needs to be global
items = list()

def getConstlist():
    simfiles = getSimfiles(constants.simdir)
    consttocheck = [(filename, utils.readConst(constants.simdir, filename)) for filename in simfiles]
    constlist = []
    for constfile, const in consttocheck:
        constfile = os.path.join(constants.simdir, constfile)
        for c in utils.unravel(const):
            resultsdir = os.path.join(constants.resultspath, c["name"])
            if c.get("disable_checking"):
                constlist.append(c)
            elif verifySimulationNecessary(constfile, resultsdir):
                constlist.append(c)
    return constlist

def prepareSim():
    utils.setup_logging_base()
    constlist = getConstlist()
    
    for const in constlist:
        info(const["name"])
    
    info("Creating final statistics for a total of %s simulations" % len(constlist))
    count = 0
    for const in constlist:
        try:
            count += 1
            info("Running final statistics only for %s:\t %s out of %s" % (const["name"], count, len(constlist))) 
            #run the simulation
            mySim = Simulation(const, noDelete=True)
            ds = classDataset.load(Dataset.AMOEBOID, mySim.resultsdir, fileprefix="A", dt=const["dt"])
            setattr(mySim, "dsA", ds)
            #run analysis algorithms
            mySim.analyse()
        except:
            error(sys.exc_info()[0])
            error("Problems when analyzing %s" % const["name"])
            error(traceback.format_exc())
    info("All simulations have been run.")

    for i, const in enumerate(items):
        functions = []
        if const["create_path_plot"]:
            functions.append(graphics.create_path_plot)
        if const["create_video_directly"]:
            functions.append(graphics.create_video)
            
        info("Creating graphic output for %s, %s out of %s" % (const["name"], i+1, len(items)))
        graphics.writeFrames(const, output_func=functions)
    
if __name__ == "__main__":
    prepareSim()