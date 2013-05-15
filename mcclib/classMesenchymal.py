import os
import constants, utils, graphutils

class Mesenchymal():
    '''
    Represents an agent that can use his energy to degrade walls..
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
        Mesenchymal.eat = graphutils.loadImage(eatpath)
        Mesenchymal.eat = utils.scaleToMax(constants.wallconst, Mesenchymal.eat)
        Mesenchymal.degrad_rate = const["zeta"]
        Mesenchymal.safety_factor = const["safety_factor"]
        Mesenchymal.classInitialized = True
    
    @staticmethod
    def getMaxEatingDistance(density):
        shape = tuple(map(round, (1/density) * Mesenchymal.eat.shape))
        return 0.5 * Mesenchymal.safety_factor * max(shape)
