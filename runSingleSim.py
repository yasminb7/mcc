'''
Created on 09.10.2012

@author: frederic
'''
from mcclib import utils
import graphics
from mcclib.classSimulation import Simulation

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
    
if __name__ == "__main__":
    simfile = "test-mesenchymal.py"
    prepareSim()