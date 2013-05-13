import sys, os, traceback

import graphics
import mcclib.utils as utils
import mcclib.constants as constants
from mcclib.utils import getSimfiles, verifySimulationNecessary
from mcclib.classSimulation import Simulation
from mcclib.utils import info, error

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
    
    info("Running total of %s simulations" % len(constlist))
    count = 0
    for const in constlist:
        try:
            count += 1
            info("Running simulations for %s:\t %s out of %s" % (const["name"], count, len(constlist))) 
            #run the simulation
            mySim = Simulation(const)
            mySim.run()
            #run analysis algorithms
            mySim.analyse()
            if (const["create_video_directly"] or const["create_path_plot"]) and not const["save_finalstats_only"]:
                info("Item added: %s" % const["name"])
                items.append(const)
        except KeyboardInterrupt:
            info("Aborting by user request.")
            mySim.cleanUp()
            sys.exit(1)
        except:
            error("Problems with simulation %s" % const["name"])
            error(traceback.format_exc())
    info("All simulations have been run.")

    for i, const in enumerate(items):
        functions = []
        if const["create_path_plot"]:
            functions.append(graphics.create_path_plot)
        if const["create_video_directly"]:
            functions.append(graphics.create_plots_ds)
            
        info("Creating graphic output for %s, %s out of %s" % (const["name"], i+1, len(items)))
        graphics.writeFrames(const, output_func=functions)
    
if __name__ == "__main__":
    prepareSim()