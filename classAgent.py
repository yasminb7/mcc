'''
Created on Jul 18, 2012

@author: frederic
'''
import scipy as sp
import math
from utils import norm, normsq, info
from sim import States
from classMaze import arrayIndexFromPos

class Agent(object):
    '''
    Base class for Amoeboid and Mesenchymal agents.
    '''
    
    def __init__(self, i, pos, vel, E, const, m = None, state = States.MOVING, statechange=0.0, period = None, delay = None):
        '''
        Constructor
        '''
        self.const = const
        self.i = i
        self.pos = pos
        self.vel = vel
        self.state = state
        self.statechange = statechange
        self.E = E
        self.m = const["mass"] if m is None else m
        self.orientationperiod = const["orientationperiod"] if period is None else period
        #self.lastorientation = - period * sp.random.random() if period is not None else 0.0
        self.orientationdelay = const["orientationdelay"] if delay is None else delay
        #TODO: should this really be standard_normal() ?
        self.direction_angle = 2 * sp.pi * sp.random.standard_normal()
        self.eating = False
        self.F_inter = sp.zeros(2)
        
#    def v(self, iii):
#        try:
#            val = math.sqrt(sp.sum(self.vel[iii]*self.vel[iii]))
#        except:
#            print "Schu hada"
#        return val
#    
#    def v2(self, iii):
#        return sp.sum(self.vel[iii]*self.vel[iii])
#    
#    def toString(self):
#        s = ""
#        s += sp.array_str(self.pos) + ";"
#        s += sp.array_str(self.vel) + ";"
#        s += sp.array_str(self.E) + ";"
#        s += self.state + ";"
#        return s

#    def interact(self, a, iii):
#        r = self.pos-a.pos
#        dist_sq = normsq(r)
#        F_temp = self.const["coupling"] * r / dist_sq
#        self.F_inter += F_temp[iii]
#        self.state = States.COLLIDING
#        a.F_inter -= F_temp[iii]
#        a.state = States.COLLIDING
#    
#    def set_direction(self, vector):
#        x = vector[0]
#        y = vector[1]
#        self.direction_angle = 0.0
#        self.direction_angle = sp.arctan2(y, x)
        #NEVER do this again!
        #if x != 0.0:
        #    self.direction_angle = sp.arctan(y/x)
    
#    def set_direction_angle(self, angle):
#        self.direction_angle = angle
#    
#    def change_direction(self, cgrad):
#        """Sets the new vector to cgrad plus some compass noise"""
#        self.set_direction(cgrad)
#        noise = self.const["compass_noise"] * sp.random.standard_normal()
#        self.direction_angle += noise
#        return self.get_direction()
#    
#    def get_direction(self):
#        alpha = self.direction_angle
#        direction = sp.array([math.cos(alpha), math.sin(alpha)])
#        return direction
    
#    def evolve_orientation(self, dt, cgrad):
#        reorient = self.lastorientation >= self.orientationperiod and self.lastorientation < self.orientationperiod + self.orientationdelay
#        moveagain = self.lastorientation < 0.0 or self.lastorientation >= self.orientationperiod+self.orientationdelay
#        direction = self.get_direction()
#        if reorient:
#            direction = sp.zeros(2)
#            self.state = States.ORIENTING
#        if moveagain:
#            direction = self.change_direction(cgrad)
#            self.lastorientation = 0.0
#        self.lastorientation += dt
#        return direction, 0.0 if reorient else 1.0
