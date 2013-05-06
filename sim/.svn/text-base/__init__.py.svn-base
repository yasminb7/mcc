'''
Created on 09.10.2012

@author: frederic
'''
import string

def enum(**enums):
    return type('Enum', (), enums)

States = enum(MOVING='g', ORIENTING='b', BLOCKED='r', DEGRADING='y', CLOSE='c', COLLIDING='m')

def getConst(const, vartuple, exclude=None):
    factors = const["factors"]
    assert len(vartuple)==len(factors)
    newconst = const.copy()
    if exclude=="repetitions" and "repetitions" in factors:
        newconst["name"] = string.replace(newconst["name"], "_r%s", "")
        vartuple = tuple([vartuple[i] for i in range(len(vartuple)-1)])
    newname = newconst["name"] % (vartuple)
    for i, n in enumerate(vartuple):
        newconst[factors[i]] = vartuple[i]
    newconst["name"] = newname
    return newconst