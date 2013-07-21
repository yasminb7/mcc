"""
Compares a given `simfile` to prototype.py (which is assumed to be correct) and
lets you know if some parameters in the simfile are missing or in excess.
"""

import mcclib.constants as constants
import mcclib.utils as utils
import analytics
import math


def main():
    basedir = "/media/Seagate Expansion Drive/mcc/sim"
    simfile = "_iexplore.py" #constants.currentsim
    protosim = "prototype.py"
    _const = utils.readConst(basedir, simfile)
    if _const is None:
        print "Could not load sim file"
        return
    constlist = utils.unravel(_const)
    
    _prototype = utils.readConst(constants.simdir, protosim)
    if _prototype is None:
        print "Could not load prototype"
        return
    prototype = utils.unravel(_prototype)[0] 
    
    for const in constlist:
        for k in prototype.keys():
            if k not in const:
                print "%s does not exist" % k
        for k in const.keys():
            if k not in prototype:
                print "%s is not needed" % k
        break




if __name__ == '__main__':
    main()    
