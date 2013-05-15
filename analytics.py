import numpy as sp
import math
import mcclib.utils as utils
import mcclib.graphutils as graphutils
import mcclib.constants as constants

def estimateAvgEnergy(const, ECMDensity, N_a, N_m):
    e_d = equilEnergyDegrading(const)
    e_p = equilEnergyPropulsing(const)
    N = N_a + N_m
    rho = ECMDensity
    return ( N_m*( e_p + rho*(e_d-e_p) ) + N_a*e_p)/N

def getECMDensity(maze):
    return sp.mean(maze)

def maxIdleEnergy(const):
    q = const["q"]
    delta = const["delta"]
    return q/delta 

def equilEnergyDegrading(const):
    gamma = const["gamma_m"]
    eta = const["eta"]
    return gamma/eta*equilSpeedDegrading(const)

def equilEnergyPropulsing(const):
    gamma = const["gamma_a"]
    eta = const["eta"]
    return gamma/eta*equilSpeedPropulsing(const)

def equilSpeedDegrading(const, q):
    return equilSpeed(const, const["gamma_m"], const["delta"], const["zeta"], q)

def equilSpeedPropulsing(const, q):
    return equilSpeed(const, const["gamma_a"], const["delta"], 0.0, q)

def equilSpeed(const, gamma, delta, zeta, q, plus=True):
    eta = const["eta"]
    discr = (delta+zeta)**2 / (4*eta*eta) + q/gamma
    if plus:
        return - (delta+zeta)/(2*eta) + math.sqrt( discr )
    else:
        return - (delta+zeta)/(2*eta) - math.sqrt( discr )

def minPathLength(const):
    goal = sp.array(const["gradientcenter"])
    initpos = sp.array(const["initial_position"])
    v = goal-initpos
    return sp.sqrt( sp.sum(v*v) )

if __name__=="__main__":
    simfile = constants.currentsim
    _const = utils.readConst(constants.confpackage, simfile)
    constlist = utils.unravel(_const)
    constlist = utils.applyFilter(constlist, "repetitions", [0])
    for const in constlist:
        mazepath = const["maze"]
        maze = graphutils.loadImage(mazepath)
        ECMdensity = getECMDensity(maze)
        print "%s:\t ECM density is %s" % (const["name"], ECMdensity)