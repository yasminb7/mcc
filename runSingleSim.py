"""
Runs simulations in a way similar to ``main.py``, but more aggressive:
It creates all possible simulations given by the variable `simfile` and runs them, no questions asked.

More specifically, unlike ``main.py``, it:
* bypasses timestamp checks and
* runs only a specific sim-file given (instead of all files in ``sim/``). 
"""

from mcclib import utils, constants
from main import prepareSim
    
if __name__ == "__main__":
    simfile = "_free-ci.py"
    constFromFile = utils.readConst(constants.simdir, simfile)
    if constFromFile is not None:
        constlist = utils.unravel(constFromFile)
        prepareSim(constlist)
    else:
        print "Could not load file %s" % simfile
