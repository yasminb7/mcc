'''
Created on 28.01.2013

@author: frederic
'''
import constants
import utils
import analytics
import math

if __name__ == '__main__':
    simfile = constants.currentsim
    _const = utils.readConst(constants.simdir, simfile)
    constlist = utils.unravel(_const)
    
    for const in constlist:
        print const["name"]
        e_p = analytics.equilEnergyPropulsing(const)
        e_d = analytics.equilEnergyDegrading(const)
        print "Energy"
        print "\twhen propulsing\t%0.2f" % e_p
        print "\twhen degrading\t%0.2f" % e_d
        print "\twhen idle\t%0.2f" % analytics.maxIdleEnergy(const) 
        v_p = analytics.equilSpeedPropulsing(const)
        v_d = analytics.equilSpeedDegrading(const)
        print "Speed"
        print "\twhen propulsing\t%0.2f" % v_p
        print "\twhen degrading\t%0.2f" % v_d
        print "\tratio\t\t%0.2f" % (v_p/v_d)
        r = const["radius"] #r should be in mu m
        rho = 1e-6 #mu g per mu m^3
        mass_th = 4.0/3.0 * math.pi * (r*r*r) * rho
        print "Mass should be\t%0.2e mu g" % mass_th
        print "Mass is %0.2e" % const["mass"]
        
    
    