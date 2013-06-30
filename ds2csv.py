"""
Takes a stored :py:class:`.Dataset` in given folders and copies it
to the *CSV* format for easier analysis with Excel, MATLAB etc.
The output files are saved in the same folders.

Please specify the folders from which you want to take the data in the :py:func:`main` function.

**It's easy to adapt this script to print other values like agent states.**
"""

import os
import csv

from mcclib.classDataset import load
from mcclib.classDataset import Dataset
import mcclib.constants as constants

def main():
    """Just calls :py:func:`saveToCSV` for a given list of folders."""
    folders = [
               "maze-easy-ar_pM0.1_q1.0_r0",
               ]
    for folder in folders:
        saveToCSV(folder)

def saveToCSV(folder):
    """Calls the functions :py:func:`writePositions`, :py:func:`writeVelocities` and
    :py:func:`writeEnergies` in order to write the respective CSV files from the given dataset."""
    resultsbase = constants.resultspath
    path = os.path.join(resultsbase, folder)
    
    ds = load(Dataset.ARRAYS, path, "A_", 0.1, readOnly=True)
    
    pathtocsv_pos = os.path.join(path, '%s-positions.csv' % folder)
    writePositions(ds, pathtocsv_pos)
    
    pathtocsv_vel = os.path.join(path, '%s-velocities.csv' % folder)
    writeVelocities(ds, pathtocsv_vel)
    
    pathtocsv_en = os.path.join(path, '%s-energies.csv' % folder)
    writeEnergies(ds, pathtocsv_en)
    
    print "Wrote %s" % folder
    print "Done."

def writePositions(ds, pathtocsv):
    """
    Writes a csv file containing as columns first the time
    and then x and y values of position for each agent.
    """
    with open(pathtocsv, 'wb') as _file:
        mydialect = csv.excel
        csvfile = csv.writer(_file, dialect=mydialect)
        agentIndices = xrange(ds.N_agents)
        writeIxNx2(agentIndices, ds.times, ds.positions, csvfile)

def writeVelocities(ds, pathtocsv):
    """
    Essentially the same as :py:func:`writePositions`, except that it's for velocities.
    """ 
    with open(pathtocsv, 'wb') as _file:
        mydialect = csv.excel
        csvfile = csv.writer(_file, dialect=mydialect)
        agentIndices = xrange(ds.N_agents)
        writeIxNx2(agentIndices, ds.times, ds.velocities, csvfile)
        
def writeEnergies(ds, pathtocsv):
    """
    Essentially the same as :py:func:`writePositions`, except that it's for energies
    (and therefore only dimension per agent and timestep).
    """ 
    with open(pathtocsv, 'wb') as _file:
        mydialect = csv.excel
        csvfile = csv.writer(_file, dialect=mydialect)
        agentIndices = xrange(ds.N_agents)
        writeIxN(agentIndices, ds.times, ds.energies, csvfile)

def writeIxNx2(agentIndices, times, arr, csvfile):
    """Writes out the data for arrays in of the shape I*N*2, meaning `I` timesteps, `N` agents and 2 numbers per agent and timestep."""
    header = ["Time"]
    for i in agentIndices:
        header.append("Agent %d x" % (i,) )
        header.append("Agent %d y" % (i,) )
    csvfile.writerow(header)
    for i, t in enumerate(times):
        row = [ "%.3f" % t ]
        for j in agentIndices:
            row.append( "%.3f" % arr[i, j, 0] )
            row.append( "%.3f" % arr[i, j, 1] )
        csvfile.writerow( row )

def writeIxN(agentIndices, times, arr, csvfile):
    """Writes out the data for arrays in of the shape I*N, meaning `I` timesteps and `N` agents."""
    header = ["Time"]
    for i in agentIndices:
        header.append("Agent %d " % (i,) )
    csvfile.writerow(header)
    for i, t in enumerate(times):
        row = [ "%.3f" % t ]
        for j in agentIndices:
            row.append( "%.3f" % arr[i, j] )
        csvfile.writerow( row )

if __name__=="__main__":
    main()