'''
Created on 09.10.2012

@author: frederic
'''
import utils, graphics
from utils import getSimfiles, verifySimulationNecessary
from classSimulation import Simulation
import constants
from utils import info, error

def prepareSim(const):
    utils.setup_logging_base() 
    #run the simulation
    mySim = Simulation(const)
    mySim.run()
    #run analysis algorithms
    mySim.analyse()
    functions = []
    if const["create_video_directly"]:
        functions.append(graphics.create_plots_ds)
    if const["create_path_plot"]:
        functions.append(graphics.create_path_plot)
    graphics.writeFrames(const, output_func=functions)
        #videoname = "animation_%s.mpg" % const["name"]
        #path = os.path.join(cwd, constants.resultspath, const["name"])
        #graphics.write_mpg(path, videoname)
    
if __name__ == "__main__":
    simfile = "test-mesenchymal.py"
    prepareSim()