'''
Created on 14.10.2012

@author: frederic
'''

import scipy as sp
from classAgent import Agent
import classDataset
from classDataset import Dataset
from constants import simdir
import cPickle

step = 5

class lineinfo:
    def __init__(self, line):
        t, a, p, v, E, state, disregard = line.split(";")
        self.timestep = float(t)
        self.agent = int(a) 
        self.pos = sp.fromstring(p.translate(None, '[]').strip(), sep=' ')
        self.vel = sp.fromstring(v.translate(None, '[]').strip(), sep=' ')
        self.E = float(E)
        self.state = state

def recreate_sim(datafile, const):
    sim = dict()
    for line in datafile:
        inf = lineinfo(line)
        if inf.timestep not in sim.keys():
            sim[inf.timestep] = list()
        sim[inf.timestep].append(Agent(inf.pos, inf.vel, inf.E, m=const["mass"], state=inf.state))
    return sim

def getSim_old(datafilename, const):
    """Returns a dictionary with dict[time] containing the list of all agents"""
    datafile = open(datafilename, "r")
    sim = recreate_sim(datafile, const)
    datafile.close()
    return sim

def getSim(picklefilename):
    """Returns a dictionary with dict[time] containing the list of all agents"""
    picklefile = open(picklefilename, "r")
    sim = cPickle.load(picklefile)
    picklefile.close()
    return sim

def getDataset(arrays, path, dt, prefix=None):
    try:
        ds = classDataset.load(arrays, path, prefix, dt)
    except IOError:
        ds = None
        raise
    return ds