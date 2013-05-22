import os
import csv

from mcclib.classDataset import load
from mcclib.classDataset import Dataset
import mcclib.constants as constants


def main(folder):
    resultsbase = constants.resultspath
    path = os.path.join(resultsbase, folder)
    
    ds = load(Dataset.AMOEBOID, path, "A_", 0.1, readOnly=True)
    
    savedir = "/home/frederic/Desktop/send/"
    pathtocsv = os.path.join(savedir, '%s-positions.csv' % folder)
    
    with open(pathtocsv, 'wb') as _file:
        mydialect = csv.excel
        #mydialect.delimiter = "\t"
        csvfile = csv.writer(_file, dialect=mydialect)
        agentIndices = xrange(ds.N_agents)
        header = ["Time"]
        for i in agentIndices:
            header.append("Agent %d x" % (i,) )
            header.append("Agent %d y" % (i,) )
        csvfile.writerow(header)
        for i, t in enumerate(ds.times):
            positions = [ "%.3f" % t ]
            for j in agentIndices:
                positions.append( "%.3f" % ds.positions[i, j, 0] )
                positions.append( "%.3f" % ds.positions[i, j, 1] )
            csvfile.writerow( positions )
    print "Wrote %s" % folder
    print "Done."

if __name__=="__main__":
    folders = [
               "maze-easy-ar_pM0.1_q1.0_r0",
               "maze-easy-ar_pM0.9_q1.0_r0",
               "maze-easy-ar_pM0.2_q1.0_r0",
               "maze-easy-ar_pM0.8_q1.0_r0",
               "maze-easy-ar_pM0.3_q0.7_r0",
               "maze-easy-ar_pM0.7_q0.7_r0"
               ]
    for folder in folders:
        main(folder)
