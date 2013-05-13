from mcclib import utils, constants
import os

sims = [constants.currentsim] 

for currentsim in sims:    
    const = utils.readConst(constants.simdir, currentsim)
    constlist = utils.unravel(const)
    cwd = os.getcwd()
    print "Trying to update %s folders" % len(constlist)    
    for i, c in enumerate(constlist):
        path = os.path.join(cwd, constants.resultspath, c["name"])
        print "%03d/%s: Updating %s:" % (i+1, len(constlist), c["name"]), 
        updated = utils.updateTimestamp(path)
        if updated:
            print "Done."
        else:
            print
print "Done."
