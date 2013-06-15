import os, math, string, csv, itertools
import numpy as np
import sim

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

def getLine(folder, pM):
    resultsbase = constants.resultspath
    path = os.path.join(resultsbase, folder)

    ds = load(Dataset.AMOEBOID, path, "A_", 0.1, readOnly=True)
    traj_a = np.array(ds.positions)
    
    consider = np.logical_or(ds.states==sim.States.MOVING, ds.states==sim.States.ORIENTING)
    velocities = np.ma.empty_like(ds.velocities)
    velocities[:,:,0] = np.ma.array(ds.velocities[:,:,0], mask=~consider)
    velocities[:,:,1] = np.ma.array(ds.velocities[:,:,1], mask=~consider)
    
    v = np.sqrt( np.sum(velocities*velocities, axis=2) )
    pathlengths = np.sum(v*ds.dt, axis=0)
    
    N_mesenchymal = np.count_nonzero(ds.types==constants.TYPE_MESENCHYMAL)
    N_amoeboid = np.count_nonzero(ds.types==constants.TYPE_AMOEBOID)

    goal = np.array((800,800))
    dt = 0.1
    distFromGoal = np.array( goal - ds.positions )
    distSq = np.sum( distFromGoal*distFromGoal, axis=2)
    hasArrived = distSq < 100**2
    timeToTarget = []
    for i in xrange(N_agents):
        iCount = np.count_nonzero(hasArrived[:,i]==False) 
        timeToTarget.append( (dt * iCount) if iCount != 25000 else "")
    
    displacement = ds.positions[-1] - ds.positions[0]
    straightPath = goal - ds.positions[0]
    straightPath_n = np.sqrt( np.sum(straightPath*straightPath, axis=1))
    straightPath_u = np.zeros_like(straightPath)
    for i in np.nonzero(straightPath_n):
        straightPath_u[i] = straightPath[i] / (straightPath_n)[i][:,np.newaxis]
    #implement the dot product between straightPath_u and displacement
    ci = np.sum( straightPath_u*displacement, axis=1)/pathlengths
    
    successful = []
    for i in range(0, ds.N_agents):
        bSuccess = isSuccessful(i, traj_a) 
        successful.append( 1 if bSuccess else 0)

    succ = np.array(successful)
    success_rate = np.mean(successful)
    
    agentData = dict()
    
    for i in range(ds.N_agents):
        line = []
        line.append(agenttype(ds.types[i]))
        line.append(successful[i])
        line.append(timeToTarget[i])
        line.append(ci[i])
        agentData[i] = line

    return agentData
    

def agenttype(t):
    if t == 0:
        return "A"
    elif t == 1:
        return "M"

def writeAgentData(r, ciFile, agentData):
    for agent, line in agentData.iteritems():
        lineToPrint = [r, agent]
        lineToPrint.extend(line)
        ciFile.writerow(lineToPrint)

if __name__=="__main__":
    folders = [
               ("maze-easy-ar_pM0.0_q1.0_r%d", 0.0),
#               ("maze-easy-ar_pM0.1_q1.0_r%d", 0.1),
#               ("maze-easy-ar_pM0.2_q1.0_r%d", 0.2),
#               ("maze-easy-ar_pM0.3_q1.0_r%d", 0.3),
#               ("maze-easy-ar_pM0.4_q1.0_r%d", 0.4),
#               ("maze-easy-ar_pM0.5_q1.0_r%d", 0.5),
#               ("maze-easy-ar_pM0.6_q1.0_r%d", 0.6),
#               ("maze-easy-ar_pM0.7_q1.0_r%d", 0.7),
#               ("maze-easy-ar_pM0.8_q1.0_r%d", 0.8),
#               ("maze-easy-ar_pM0.9_q1.0_r%d", 0.9),
#               ("maze-easy-ar_pM1.0_q1.0_r%d", 1.0),
               ]
    N_agents = 50
    for folder, pM in folders:
        foldername = string.replace(folder, "_r%d", "")
        with open("%s_ci.csv" % foldername, "w+") as ciFile_:
            ciFile = csv.writer(ciFile_, dialect=csv.excel)
            title1 = ["Repetition #", "Agent #", "Type", "Successful", "Time to target", "CI"]
            #agentDescriptions1 = [ 4*["%d" % i] for i in range(0, N_agents)]
            #title1.extend(itertools.chain(*agentDescriptions1))
            #title2 = ["Variable"]
            #agentDescriptions2 = [["Type", "successful", "Time to target", "CI"] for i in range(0, N_agents)]
            #title2.extend(itertools.chain(*agentDescriptions2))
            ciFile.writerow(title1)
            #ciFile.writerow(title2)
            for r in range(0, 10):
                repetitionF = folder % r
                agentData = getLine(repetitionF, pM)
                writeAgentData(r, ciFile, agentData)
