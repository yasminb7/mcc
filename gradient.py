'''
Created on Jul 29, 2012

@author: frederic
'''
import scipy as sp
from scipy.linalg import norm

def concentration(i, j, gradientcenter):
    c = gradientcenter
    field = (i-c[0]) * (i-c[0]) + (j-c[1]) * (j-c[1])
    return -1.0 * field

def nullconcentration(i, j):
    return sp.cast['f'](0.0)

#def gradient(pos, c = sp.array([80.0, 80.0])):
#    #g = -0.2 * pos*pos*pos
#    n = norm(c-pos)
#    lower_limit = 0.000001
#    if n<lower_limit:
#        n=lower_limit
#    g = (c-pos)/n
#    return g
#
#def gradientx(pos):
#    return gradient(pos[0])
#
#def gradienty(pos):
#    return gradient(pos[1])
#
#def unitgradient(pos):
#    g = gradient(pos)
#    n = norm(g)
#    return g/n