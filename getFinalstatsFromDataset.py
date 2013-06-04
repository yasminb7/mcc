import sys, os, traceback
import graphics
from mcclib import utils, constants, classDataset
from mcclib.classSimulation import Simulation
from mcclib.classDataset import Dataset
from mcclib.utils import info, error
import mcclib.statutils as statutils

def prepareSim():
    utils.setup_logging_base()
    
    simfile = "maze-easy-ar.py"
    const = utils.readConst(constants.simdir, simfile)
    print const
    
    folders = ["maze-easy-ar_pM0.1_q1.0_r0",
               "maze-easy-ar_pM0.9_q1.0_r0",
               "maze-easy-ar_pM0.2_q1.0_r0",
               "maze-easy-ar_pM0.8_q1.0_r0",
               "maze-easy-ar_pM0.3_q0.7_r0",
               "maze-easy-ar_pM0.7_q0.7_r0]"]
    
    f = folders[4]
    resultsbasepath = os.path.join(constants.resultspath, f)
    
    info("Creating final statistics for %s" % f)
    
    try:
        mySim = Simulation(const, noDelete=True)
        ds = classDataset.load(Dataset.AMOEBOID, mySim.resultsdir, fileprefix="A", dt=const["dt"])
        setattr(mySim, "dsA", ds)
        finalstats = statutils.finalstats(ds, (800,800), 100)
        print finalstats
    except:
        error(sys.exc_info()[0])
        error("Problems when analyzing %s" % const["name"])
        error(traceback.format_exc())
    
if __name__ == "__main__":
    prepareSim()