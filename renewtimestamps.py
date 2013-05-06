'''
Created on 25.11.2012

@author: frederic
'''
import utils, constants
import os

sims = [
        #"amoeboids-compass-noise.py",
        #"effect-of-mesenchymals2.py",
        #"effect-of-q.py",
        #"free-movement.py",
        #"one-free.py",
        #"onesim-test.py",
        #"yay.py"
        constants.currentsim
        ] 

for currentsim in sims:    
    const = utils.readConst(constants.simdir, currentsim)
    constlist = utils.unravel(const)
    cwd = os.getcwd()    
    for c in constlist:
        path = os.path.join(cwd, constants.resultspath, c["name"])
        print "Updating %s" % c["name"]
        utils.updateTimestamp(path)
print "Done."
