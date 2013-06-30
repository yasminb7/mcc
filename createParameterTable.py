"""
Creates a csv file (parameters.csv) that contains all the parameters for a given `sim` file.
"""

import csv
from mcclib import constants, utils


sim = constants.currentsim
#alternatively, you can directly specify a name as in 
#sim = "prototype.py"

const = utils.readConst(constants.simdir, sim)
#constlist = utils.unravel(const)

outFileName = "parameters.csv"

with open(outFileName, "wb") as outfile:
    outCSV = csv.writer(outfile, dialect=csv.excel)
    for name, value in const.iteritems():
        outCSV.writerow([name, value])