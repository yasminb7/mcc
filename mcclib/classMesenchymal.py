import os
import constants, utils

class Mesenchymal():
    '''
    Holds a few variables and a method that have to do with mesenchymals.
    Does **not** in any way represent a mesenchymal agent.
    It's contents could very well be put directly in :py:class:`.Simulation`. 
    '''
    
    classInitialized = False
    eat = None
    z = 0.0
    degrad_rate = 0.0
    safety_factor = 1.0
        
    @staticmethod
    def classInit(const, basepath=None):
        if basepath is None:
            basepath = os.getcwd()
        eatpath = os.path.join(basepath, constants.resourcespath, const["eatshape"])
        Mesenchymal.eat = utils.loadImage(eatpath)
        Mesenchymal.eat = utils.scaleToMax(constants.wallconst, Mesenchymal.eat)
        Mesenchymal.degrad_rate = const["zeta"]
        Mesenchymal.safety_factor = const["safety_factor"]
        Mesenchymal.classInitialized = True
    
    @staticmethod
    def getMaxEatingDistance(density):
        shape = tuple(map(round, (1/density) * Mesenchymal.eat.shape))
        return 0.5 * Mesenchymal.safety_factor * max(shape)
