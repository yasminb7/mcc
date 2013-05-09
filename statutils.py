'''
Created on 14.10.2012

@author: frederic
'''

import scipy as sp
import scipy.stats as stats
import cPickle, os
import utils
import constants
import sim


def hist(values, bins=10, limits=None):
    hist = stats.histogram(values, numbins=bins, defaultlimits=limits)
    return hist

def avg_vel(velocities):
    """Returns velocity averaged over agents AND time"""
    if velocities.size != 0:
        velocities = sp.sqrt( sp.sum(velocities*velocities, axis=2) )
        return sp.mean(velocities)
    else:
        return None

def getDistances(positions, goal):
    """Return an array containing the distances of all agents from the goal."""
    last_axis = len(positions.shape)-1
    dist_v = positions-goal 
    dist = sp.sqrt( sp.sum(dist_v*dist_v, axis=last_axis) )
    return dist

def getSuccessrates(dist, success_radius):
    """Returns success rates averaged over agents (axis=1) but not averaged over time."""
    last_axis = len(dist.shape)-1
    succeeded = sp.zeros_like(dist)
    succeeded[dist < success_radius] = 1.0
    return sp.mean(succeeded, axis=last_axis)

def getAvgTimeToTarget(dist, success_radius, dt):
    migrating = dist >= success_radius
    succeeded = migrating[-1]==False
    if succeeded.any():
        return None
    times = []
    for idxA in succeeded.nonzero()[0]:
        t = getTimeToTarget(migrating[:,idxA], dt)
        times.append(t)
    times = sp.fromiter(times)
    return sp.mean(times)

def getTimeToTarget(migrating, dt):
    timesteps = sp.count_nonzero(migrating)
    return timesteps/dt

def avgEnergyForAgent(energies, states):
    """Returns the average energy over time for a single agent.
    The only states that are taken into account are MOVING and ORIENTING.
    """
    includeStep = sp.logical_or(states==sim.States.MOVING, states==sim.States.ORIENTING)
    avgEnergy = sp.mean(energies[includeStep])
    return avgEnergy

def doCombinations(stats, varname, data, amoeboids, mesenchymals, successful, function=sp.mean, axis=None):
    """Apply a given function to all elements of a given array, and do so for all desired combinations.
    The combinations are for example "all agents that are amoeboid and were successful.
    Note that some of these results may make little sense because too few agents fit the criteria.
    """
    a_s = sp.logical_and(amoeboids, successful)
    m_s = sp.logical_and(mesenchymals, successful)
    a_us = sp.logical_and(amoeboids, ~successful)
    m_us = sp.logical_and(mesenchymals, ~successful)
    combinations = {
                    "" : sp.ones_like(data, dtype=sp.bool_),
                    "_a" : amoeboids,
                    "_m" : mesenchymals,
                    "_a_s" : a_s,
                    "_m_s" : m_s,
                    "_a_us" : a_us,
                    "_m_us" : m_us,
                    }
    for suffix, selector in combinations.iteritems():
        if axis is None:
            myselector = sp.s_[selector]
        elif axis==1:
            myselector = sp.s_[:,selector]
        value = function( data[myselector] ) if selector.any() else None
        if value is not None and type(value)!=float:
            value = float(value)
        stats[varname + suffix] = value 
    return stats

def finalstats(ds, goal, success_radius):
    """Calculate some of the statistics that we're interested in at the end of a simulation."""
    stats = {}
    amoeboids = ds.types==ds.is_amoeboid
    mesenchymals = ds.types==ds.is_mesenchymal
    stats["time"] = ds.times[-1]

    dist = getDistances(ds.positions[-1], goal)
    successful = dist < success_radius
    stats = doCombinations(stats, "distance_from_goal", dist, amoeboids, mesenchymals, successful)
    
    stats["success_ratio"] = sp.mean( getSuccessrates(dist, success_radius) )
    stats["success_ratio_a"] = sp.mean( getSuccessrates(dist[amoeboids], success_radius) ) if amoeboids.any()==True else None
    stats["success_ratio_m"] = sp.mean( getSuccessrates(dist[mesenchymals], success_radius) ) if mesenchymals.any()==True else None
    
    consider = sp.logical_or(ds.states==sim.States.MOVING, ds.states==sim.States.ORIENTING)
    velocities = sp.ma.empty_like(ds.velocities)
    velocities[:,:,0] = sp.ma.array(ds.velocities[:,:,0], mask=~consider)
    velocities[:,:,1] = sp.ma.array(ds.velocities[:,:,1], mask=~consider)
    
    v = sp.sqrt( sp.sum(velocities*velocities, axis=2) )
    pathlengths = sp.sum(v*ds.dt, axis=0)
    stats = doCombinations(stats, "avg_path_length", pathlengths, amoeboids, mesenchymals, successful)
    stats = doCombinations(stats, "avg_path_length_err", pathlengths, amoeboids, mesenchymals, successful, function=sp.std)
    stats = doCombinations(stats, "avg_vel", velocities, amoeboids, mesenchymals, successful)
    
    energies = sp.ma.array(ds.energies, mask=~consider)
    stats = doCombinations(stats, "avg_en", energies, amoeboids, mesenchymals, successful, axis=1)
    
    displacement = ds.positions[-1] - ds.positions[0]
    straightPath = goal-ds.positions[0]
    straightPath_n = sp.sqrt( sp.sum(straightPath*straightPath, axis=1))
    straightPath_u = sp.zeros_like(straightPath)
    for i in sp.nonzero(straightPath_n):
        straightPath_u[i] = straightPath[i] / (straightPath_n)[i][:,sp.newaxis]
    #implement the dot product between straightPath_u and displacement
    ci = sp.sum(straightPath_u*displacement/pathlengths[:,sp.newaxis], axis=1)
    stats = doCombinations(stats, "avg_ci", ci, amoeboids, mesenchymals, successful)
    
    return stats

def dicAsText(dic):
    """Take a given Python dictionary and return it as a string."""
    text = ""
    for k, v in sorted(dic.iteritems()):
        text += "%s:\t%s\n" % (k, v)
    return text
    
def savefinalstats(picklepath, textpath, const, dataset):
    """Calculate the finalstats and save them once as a pickle file and once as a text file."""
    stats = finalstats(dataset, const["gradientcenter"], const["success_radius"])
    with open(picklepath, "w") as statfile:
        cPickle.dump(stats, statfile)
    with open(textpath, "w") as statfile:
        statfile.write(dicAsText(stats))

def saveConst(const, filename):
    with open(filename, "w") as constfile:
        constfile.write( dicAsText(const) )

def readfinalstats(finalstatspath):
    with open(finalstatspath, "r") as statfile:
        stats = cPickle.load(statfile)
    return stats

def excludeItem(lst, exclude=["repetitions"]):
    newlst = list()
    for item in lst:
        if item not in exclude:
            newlst.append(item)
    return newlst

def clearStatistics(folder):
    """Remove some files created by statistics.py in a given folder."""
    currentcwd = os.getcwd()
    os.chdir(folder)
    files = ["avg_energy.png", constants.avg_en_vel_filename, constants.avg_dist_filename]
    for f in files:
        path = os.path.join(folder, f)
        utils.deleteFile(path)
    os.chdir(currentcwd)
