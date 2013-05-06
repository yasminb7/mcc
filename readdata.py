'''
Created on 14.10.2012

@author: frederic
'''

import classDataset

def getDataset(arrays, path, dt, prefix=None):
    try:
        ds = classDataset.load(arrays, path, prefix, dt)
    except IOError:
        ds = None
        raise
    return ds