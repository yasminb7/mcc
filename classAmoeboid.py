'''
Created on 06.11.2012

@author: frederic
'''
import scipy as sp
import os
import utils, constants
#from utils import normsq
#from sim import States
#from classAgent import Agent
#import classMaze

class Amoeboid():
    '''
    What's special about the Amoeboids?
    '''
    
    classInitialized = False
    aura = None
    y = None
    safety_factor = 1.0
    
#    def __init__(self, i, pos, vel, E, state, statechange, const, m = None, period = None, delay = None):
#        if Amoeboid.classInitialized==False:
#            Amoeboid.classInit(const)
#        Agent.__init__(self, i, pos, vel, E, const, m, state, statechange=statechange, period=period, delay=delay)
    
#    @staticmethod
#    def classInit(const, basepath=None):
#        if basepath is None:
#            basepath = os.getcwd()
#        aurapath = os.path.join(basepath, constants.resourcespath, const["aura"])
#        Amoeboid.aura = utils.loadImage(aurapath)
#        Amoeboid.y = const["y"]
#        Amoeboid.safety_factor = const["safety_factor"]
#        Amoeboid.classInitialized = True
#
#    @staticmethod
#    def getMaxFeelingDistance(density):
#        shape = tuple(map(round, (1/density) * Amoeboid.aura.shape))
#        return 0.5 * Amoeboid.safety_factor * max(shape)
#    
#    def feelaround(self, myMaze, posidx):
#        wx = Amoeboid.aura.shape[0]
#        wy = Amoeboid.aura.shape[1]
#        lowerx = posidx[0] - wx / 2
#        lowery = posidx[1] - wy / 2
#        window = sp.s_[lowerx:lowerx+wx, lowery:lowery+wy]
#        overlap = sp.mean(myMaze.data[window]*Amoeboid.aura)
#        return overlap * Amoeboid.y
