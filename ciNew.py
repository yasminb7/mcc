import os
import math
import csv
import numpy as np

from mcclib.classDataset import load
from mcclib.classDataset import Dataset
import mcclib.constants as constants

def isSuccessful(agentIndex, positions, goal=(800, 800), successRadius=100.0):
    lastX = positions[-1, agentIndex, 0]
    lastY = positions[-1, agentIndex, 1]
    lastDist = math.sqrt( (goal[0]-lastX)**2 + (goal[1]-lastY)**2 )
    if lastDist <= successRadius:
        return True
    else:
        return False

def main(folder, pM, ciFile):
    resultsbase = constants.resultspath
    path = os.path.join(resultsbase, folder)
    out = "%s:\n" % folder
    print out
    csvout = [folder]
    #ciFile.write(out)
    ds = load(Dataset.AMOEBOID, path, "A_", 0.1, readOnly=True)
    traj_a = np.array(ds.positions)
    
    N_mesenchymal = int(round(pM*ds.N_agents, 0))
    N_amoeboid = ds.N_agents - N_mesenchymal
    #assert N_mesenchymal==np.count_nonzero(ds.types==ds.is_mesenchymal)
    #N_mesenchymal = np.count_nonzero(ds.types==ds.is_mesenchymal)
    #N_amoeboid = ds.N_agents - N_mesenchymal
    ci = np.zeros((N_amoeboid,));
    timesteps = 25000
    successful = []
    for i in range(N_mesenchymal, ds.N_agents):
        successful.append(isSuccessful(i, traj_a))
        x0=traj_a[0,i,0]
        y0=traj_a[0,i,1]
        x1=traj_a[-1,i,0]
        y1=traj_a[-1,i,1]
        xt=800
        yt=800
        
        s = 0.0
        for j in range(1,timesteps):
            s= s + math.sqrt( (traj_a[j,i,0]-traj_a[j-1,i,0] )**2 + (traj_a[j,i,1]-traj_a[j-1,i,1])**2 ) 

        ci[i-N_mesenchymal] = ( (x1-x0)*(xt-x0) + (y1-y0)*(yt-y0) ) / math.sqrt((xt-x0)**2+(yt-y0)**2) / s;
    succ = np.array(successful)
    success_rate = np.mean(successful)
    out = "Success rate:\t%f\n"% success_rate
    print out,
    csvout.append(success_rate)
    print "CI:\t", ci
    ci_mean = np.mean(ci)
    out = "Mean CI:\t%f\n" % ci_mean 
    print out,
    csvout.append(ci_mean)
    #ciFile.write(out)
    ci_mean_s = np.mean(ci[succ])
    out = "Mean CI successful:\t%f\n" % ci_mean_s 
    print out,
    csvout.append(ci_mean_s)
    #ciFile.write(out)
    ci_mean_us = np.mean(ci[~succ])
    out = "Mean CI unsuccessful:\t%f\n" % ci_mean_us 
    print out,
    csvout.append(ci_mean_us)
    #ciFile.write(out)
    ciFile.writerow(csvout)
    print "Done."

if __name__=="__main__":
    for r in range(0, 10):
        folders = [
                   ("maze-easy-ar_pM0.1_q1.0_r%d" % r, 0.1),
                   ("maze-easy-ar_pM0.9_q1.0_r%d" % r, 0.9),
                   ("maze-easy-ar_pM0.2_q1.0_r%d" % r, 0.2),
                   ("maze-easy-ar_pM0.8_q1.0_r%d" % r, 0.8),
                   ("maze-easy-ar_pM0.3_q0.7_r%d" % r, 0.3),
                   ("maze-easy-ar_pM0.7_q0.7_r%d" % r, 0.7),
                   ]
        with open("cis.txt", "a") as ciFile_:
            ciFile = csv.writer(ciFile_)
            for folder, pM in folders:
                main(folder, pM, ciFile)
