'''
Created on Jul 29, 2012

@author: frederic
'''

import scipy as sp
import scipy.weave as weave
from scipy.ndimage.filters import gaussian_filter
import utils
import constants
from logging import debug, info
#from classCacheLayer import CacheLayer

DIMiter = xrange(constants.DIM)
sigma = 0.6
wall = constants.wallconst
borderwall = 1.01*wall
nowall = 0.0

class PositionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
def myFilter(data, mysigma=sigma):
    return gaussian_filter(data, mysigma)

class Maze(object):
    
    def __init__(self, filename, fieldlimits, border):
        self.original = utils.loadImage(filename)
        self.original = utils.scaleToMax(wall, self.original)
        self.data = myFilter(self.original)
        assert len(self.data.shape)==2, "Maze data is not two-dimensional: %s. Are you sure it is grayscale and contains one layer and one canal only?" % str(self.data.shape)
        #self.data_grad = utils.getGradient(-self.data)
        self.data_grad = utils.getOriginalGradient(filename, maze=-self.data)
        self.border = border
        self.density = densityArray(self.data.shape, fieldlimits)
        self.buildBorders(self.original)
        self.fieldlimits = fieldlimits
        self.walls_only = sp.zeros_like(self.data)
        self.buildBorders(self.walls_only)
        self.walls_only_gr = sp.gradient(self.walls_only)
        
        self.validgrad = sp.ones(self.data.shape, dtype=sp.bool_)
        
#        self.cache = CacheLayer(self.data, self.getSingleGradient)
        
        self.minidx = 0
        self.maxidx = self.data.shape[0]-1
        
        #TODO: this probably only works well if the shape of self.data is made of multiples of 10
        #self.pixelperblock = 10 
        #self.blocksize = tuple( map(int, sp.array(self.data.shape)/self.pixelperblock) )
        #self.blocks = sp.zeros(self.blocksize, dtype=sp.bool_)
        
    #@profile
    def degrade(self, pos, eat, energy, density, dt):
        Ax = 0
        Ay = 0
        Bx = self.data.shape[0]-1
        By = self.data.shape[1]-1
        
        shape = eat.shape
        corner = [int((density[i] * pos[i]) - 0.5 * shape[i]) for i in xrange(2)]
        
        xmin = corner[0]
        xmax = corner[0]+shape[0]
        ymin = corner[1]
        ymax = corner[1]+shape[1]
        degradeRect = self.buildRect(xmin, ymin, xmax, ymax)
        clippedRect = self.clipRect(degradeRect, Ax, Ay, Bx, By)
        if clippedRect is None:
            return
        a, b = clippedRect
        #view = self.data[idx[0]:idx[0]+shape[0], idx[1]:idx[1]+shape[1]]
        view = self.data[ a[0]:b[0], a[1]:b[1] ]
        if view.shape==eat.shape:
            view -= eat
            self.validgrad[ a[0]:b[0], a[1]:b[1] ] = False
            #view[view > 0] = 0
        else:
            dax, day, dbx, dby = self.diffRect(degradeRect, clippedRect)
            eat_idx = sp.s_[dax:dbx, day:dby]
            view -= eat[eat_idx]
            self.validgrad[ dax:dbx, day:dby ] = False
            #view[view > 0] = 0
            
        #view[eat > 0]=0
        limit = 0.1
        view[view<limit]=nowall
        #view[view > nowall] = nowall
    
#    #@staticmethod
#    def getSingleGradient(self, data, posidx):
#        grads = sp.zeros(2)
#        a = 1 #window length is 2a+1
#        Ax = 0
#        Ay = 0
#        Bx = self.data.shape[0]-1
#        By = self.data.shape[1]-1
#        
#        originalRect = self.buildRect(posidx[0]-a, posidx[0]+a+1, posidx[1]-a, posidx[1]+a+1)
#        clippedRect = self.clipRect(originalRect, Ax, Ay, Bx, By)
#        if clippedRect is not None:
#            dax, day, dbx, dby = self.diffRect(originalRect, clippedRect)
#            a, b = clippedRect
#            grad = sp.gradient(data[a[0]:b[0], a[1]:b[1]])
#            grads[0] = sp.mean(grad[0])
#            grads[1] = sp.mean(grad[1]) 
#        return grads
    
#    #@profile
#    def getSingleGradient(self, posidx):
#        """posidx should have shape (N, 2)"""
#        grads = sp.zeros(2, dtype=sp.float_)
#        a = 1 #window length is 2a+1
#        window = sp.s_[ posidx[0]-a : posidx[0]+a+1, posidx[1]-a : posidx[1]+a+1 ]
#        grad = sp.gradient(-self.data[window])
#        grads[0] = sp.mean(grad[0])
#        grads[1] = sp.mean(grad[1]) 
#        return grads
    
    #@profile
    def getGradientsCpp(self, posidx):
        """posidx should have shape (N, 2)"""
        validgrad = self.validgrad
        data = self.data
        data_grad = self.data_grad
        grads = sp.empty_like(posidx, dtype=sp.float_)
        mywin = sp.zeros((2,3,3))
        code_grad = \
        """
        #line 136 "classMaze.py"
        for (uint ai=0; ai<Nposidx[0]; ai++)
        {
            int px = POSIDX2(ai,0);
            int py = POSIDX2(ai,1);
            //int valid = VALIDGRAD2(px, py);
            if (VALIDGRAD2(px, py)!=0)
            {
                GRADS2(ai, 0) = DATA_GRAD3(0, px, py);
                GRADS2(ai, 1) = DATA_GRAD3(1, px, py);
            }
            else
            {
                for (int m=-1; m<=1; m++){
                    for (int n=-1; n<=1; n++){
                        MYWIN3(0, m+1,n+1) = 0.5 * (DATA2(px+m+1,py+n  ) - DATA2(px+m-1,py+n  ));
                        MYWIN3(1, m+1,n+1) = 0.5 * (DATA2(px+m  ,py+n+1) - DATA2(px+m  ,py+n-1));
                    }
                }
                float sum0 = 0.0;
                float sum1 = 0.0;
                for (int m=0; m<=2; m++){
                    for (int n=0; n<=2; n++){
                        sum0 -= MYWIN3(0, m,n);
                        sum1 -= MYWIN3(1, m,n);
                    }
                }
                GRADS2(ai,0) = sum0/9.0;
                GRADS2(ai,1) = sum1/9.0;
                DATA_GRAD3(0, px, py) = GRADS2(ai,0);
                DATA_GRAD3(1, px, py) = GRADS2(ai,1);
                VALIDGRAD2(px, py) = Py_True;
            }
        }
        """
        weave.inline(code_grad, ["data", "mywin", "grads", "data_grad", "validgrad", "posidx"])
        return grads
    
    def getGradientsPython(self, posidx):
        """posidx should have shape (N, 2)"""
        validgrad = self.validgrad
        data = self.data
        data_grad = self.data_grad
        grads = sp.empty_like(posidx, dtype=sp.float_)
        a = 1 #window length is 2a+1
        #grad = sp.empty((2*a+1, 2*a+1))
        for i, pos in enumerate(posidx):
            #pos = posidx[i]
            posx, posy = pos
            if validgrad[posx, posy]:
                grads[i] = data_grad[:,posx, posy]
            else:
                #window = sp.s_[ posx-a : posx+a+1, posy-a : posy+a+1 ]
                #grad = sp.gradient(-self.data[window])
                window = data[posx-1 : posx+2, posy-1 : posy+2]
                grad = sp.gradient(-window)
                grads[i] = sp.mean(grad[0]), sp.mean(grad[1])
                data_grad[:, posx, posy] = grads[i]
                validgrad[posx, posy] = True
                #grads[i] = sp.mean(grad[1]) 
        return grads


    
    def clipSmallrect(self, originalBig, clippedBig, small):
        pt1o, pt1c = originalBig[0], clippedBig[0]
        cL = pt1c[0]<pt1o[0]
        cR = pt1c[0]>pt1o[0]
        cB = pt1c[1]<pt1o[1]
        cT = pt1c[1]>pt1o[1]
        return cL, cR, cB, cT
    
    def diffRect(self, old, new):
        olda, oldb = old
        newa, newb = new
        dax = newa[0]-olda[0] if newa[0]-olda[0]!=0 else None 
        day = newa[1]-olda[1] if newa[1]-olda[1]!=0 else None
        dbx = newb[0]-oldb[0] if newb[0]-oldb[0]!=0 else None
        dby = newb[1]-oldb[1] if newb[1]-oldb[1]!=0 else None
        return dax, day, dbx, dby

    #@profile            
    def update(self, updatePositions, rectangle, density, needGradient=True):
        #self.blocks.flat[:] = False
        #Updates a little larger than rectangle to ensure continuity
        Ax = 0
        Ay = 0
        Bx = self.data.shape[0]-1
        By = self.data.shape[1]-1
        
        for pos in updatePositions:
            #block = self.getBlockFromPos(pos)
            #if self.blocks[block]==False:
            if True:
                windowsize = sp.array([constants.safety_factor * s for s in rectangle.shape], dtype=sp.int64)
                
                #shape = eat.shape
                corner = [int((density[i] * pos[i]) - 0.5*windowsize[i]) for i in xrange(2)]
                xmin = corner[0]
                xmax = corner[0]+windowsize[0]
                ymin = corner[1]
                ymax = corner[1]+windowsize[1]
                
                updateRect = self.buildRect(xmin, ymin, xmax, ymax)
                clippedRect = self.clipRect(updateRect, Ax, Ay, Bx, By)
                if clippedRect is None:
                    continue
                a, b = clippedRect
                
                idx = sp.s_[a[0]:b[0], a[1]:b[1]]
                idx2 =  sp.s_[:, a[0]:b[0], a[1]:b[1]]
                
                if needGradient:
                    #info("Updating window the size of %s in position %s" % (windowsize, pos))
                    temp_data = -gaussian_filter(self.data[idx], sigma)
                    self.data_grad[idx2] = sp.array( sp.gradient( temp_data ))
                else:
                    self.data[idx] = gaussian_filter(self.data[idx], sigma)
                #self.blocks[block]=True

    def getBlockFromPos(self, pos):
        return tuple(map(int, pos/self.pixelperblock))
    
    def myClip(self, pt, Ax, Ay, Bx, By):
        x = min( max(pt[0], Ax) , Bx)
        y = min( max(pt[1], Ay) , By)
        return (x, y)

    def clipRect(self, rect, Ax, Ay, Bx, By):
        pt1 = rect[0]
        pt2 = rect[1]
        pt1_ = self.myClip(pt1, Ax, Ay, Bx, By)
        pt2_ = self.myClip(pt2, Ax, Ay, Bx, By)
        if pt1_[0]<pt2_[0] and pt1_[1]<pt2_[1]:
            return (pt1_, pt2_)
        else:
            return None
    
    def buildRect(self, xmin, ymin, xmax, ymax):
        return ((xmin, ymin), (xmax, ymax))
    
    def updateAll(self):
        temp_data = myFilter(self.data)
        newgrad = utils.getGradient(-temp_data)
        self.data_grad[0] = newgrad[0]
        self.data_grad[1] = newgrad[1]
    
    def getMaze(self):
        return self.data
    
    def getMazeGradient(self):
        return self.data_grad
    
    def getGradImageSquared(self):
        return self.data_grad[0]*self.data_grad[0] + self.data_grad[1]*self.data_grad[1]
    
    def getData(self):
        """
        Returns the maze and its gradient.
        """
        return self.data
    
    def buildBorders(self, arr, arr_gr=None):
        b_v, b_h = ( int(self.border * self.density[i]) for i in xrange(2))
        #We probably need to use a higher gradient for the wall at the border of the maze
        #in order to have a reliable gradient in the direction of the center
        arr[:b_h,:]  = borderwall
        arr[-b_h:,:] = borderwall
        arr[:,:b_v]  = borderwall
        arr[:,-b_v:] = borderwall
        if arr_gr is not None:
            for i in xrange(2):
                arr_gr[i][:b_h,:]  = self.walls_only_gr[i][:b_h,:]
                arr_gr[i][-b_h:,:] = self.walls_only_gr[i][-b_h:,:]
                arr_gr[i][:,:b_v]  = self.walls_only_gr[i][:,:b_v]
                arr_gr[i][:,-b_v:] = self.walls_only_gr[i][:,-b_v:]          
        
    def withinPlayingField(self, pos):
        return self.withinBounds(pos, self.border)
    
    def withinBounds(self, pos, margin=0):
        if pos[0]<self.fieldlimits[0]+margin or pos[0]>self.fieldlimits[1]-margin or pos[1]<self.fieldlimits[2]+margin or pos[1]>self.fieldlimits[3]-margin:
            return False
        else:
            return True

def densityArray(shape, fieldlimits):
    density = sp.empty((2,))
    for k in DIMiter:
        density[k] = shape[k]/sp.absolute(fieldlimits[2*k+1]-fieldlimits[2*k])
    return density
        
#def clamp(n, minn, maxn):
#    return max(min(maxn, n), minn)
#
#def clamp2(n, minn, maxn):
#    n = n if n>minn and n<maxn else minn if n<minn else maxn
#    return int(n)

def enclosingRect(positions):
    ll = (min(positions[:,0]), max(positions[:,0]))
    ur = (min(positions[:,1]), max(positions[:,1]))
    return (ll, ur)

def bisectCount(positions, rect):
    
    xhalf = rect[0]/2
    yhalf = rect[1]/2
    left = positions[:,0]<xhalf
    right = ~left
    lower = positions[:,1]<yhalf
    upper = ~lower
    
    LL = sp.logical_and(lower, left)
    LR = sp.logical_and(lower, right)
    UL = sp.logical_and(upper, left)
    UR = sp.logical_and(upper, right)
    
    LL_ = sp.count_nonzero(LL)
    LR_ = sp.count_nonzero(LR)
    UL_ = sp.count_nonzero(UL)
    UR_ = sp.count_nonzero(UR)
    return ([LL_, LR_, UL_, UR_], [LL, LR, UL, UR])

def arrayIndexFromPos(pos, density):
    """Returns the array index corresponding to pos as a tuple.
    pos should be a numpy array."""
    #density = sp.array((0.0, 0.0))
    #idx = sp.empty((2,))
    #for k in DIMiter:
    #    density[k] = shape[k]/sp.absolute(fieldlimits[2*k+1]-fieldlimits[2*k])
    #idx = sp.array(density * pos, dtype=sp.int0)
    return tuple(sp.array(density * pos, dtype=sp.int0))
