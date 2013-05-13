import numpy as np
import scipy.weave as weave
from scipy.ndimage.filters import gaussian_filter
import utils
import constants

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
        self.data_grad = utils.getOriginalGradient(filename, maze=-self.data)
        self.border = border
        self.density = densityArray(self.data.shape, fieldlimits)
        self.buildBorders(self.original)
        self.fieldlimits = fieldlimits
        self.walls_only = np.zeros_like(self.data)
        self.buildBorders(self.walls_only)
        self.walls_only_gr = np.gradient(self.walls_only)
        
        self.validgrad = np.ones(self.data.shape, dtype=np.bool_)
        
        self.minidx = 0
        self.maxidx = self.data.shape[0]-1
        

    def degrade(self, pos, bite, energy, density, dt):
        Ax = 0
        Ay = 0
        Bx = self.data.shape[0]-1
        By = self.data.shape[1]-1
        
        shape = bite.shape
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
        view = self.data[ a[0]:b[0], a[1]:b[1] ]
        if view.shape==bite.shape:
            view -= bite
            self.validgrad[ a[0]:b[0], a[1]:b[1] ] = False
        else:
            dax, day, dbx, dby = self.diffRect(degradeRect, clippedRect)
            eat_idx = np.s_[dax:dbx, day:dby]
            view -= bite[eat_idx]
            self.validgrad[ dax:dbx, day:dby ] = False
            
        limit = 0.1
        view[view<limit]=nowall
    
    #@profile
    def getGradientsCpp(self, posidx):
        """posidx should have shape (N, 2)"""
        validgrad = self.validgrad #@UnusedVariable
        data = self.data #@UnusedVariable
        data_grad = self.data_grad #@UnusedVariable
        grads = np.empty_like(posidx, dtype=np.float_)
        mywin = np.zeros((2,3,3)) #@UnusedVariable
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
        grads = np.empty_like(posidx, dtype=np.float_)
        for i, pos in enumerate(posidx):
            posx, posy = pos
            if validgrad[posx, posy]:
                grads[i] = data_grad[:,posx, posy]
            else:
                window = data[posx-1 : posx+2, posy-1 : posy+2]
                grad = np.gradient(-window)
                grads[i] = np.mean(grad[0]), np.mean(grad[1])
                data_grad[:, posx, posy] = grads[i]
                validgrad[posx, posy] = True 
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
        #Updates a little larger than rectangle to ensure continuity
        Ax = 0
        Ay = 0
        Bx = self.data.shape[0]-1
        By = self.data.shape[1]-1
        
        for pos in updatePositions:
            if True:
                windowsize = np.array([constants.safety_factor * s for s in rectangle.shape], dtype=np.int64)
                
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
                
                idx = np.s_[a[0]:b[0], a[1]:b[1]]
                idx2 =  np.s_[:, a[0]:b[0], a[1]:b[1]]
                
                if needGradient:
                    temp_data = -gaussian_filter(self.data[idx], sigma)
                    self.data_grad[idx2] = np.array( np.gradient( temp_data ))
                else:
                    self.data[idx] = gaussian_filter(self.data[idx], sigma)
    
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
    density = np.empty((2,))
    for k in DIMiter:
        density[k] = shape[k]/np.absolute(fieldlimits[2*k+1]-fieldlimits[2*k])
    return density

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
    
    LL = np.logical_and(lower, left)
    LR = np.logical_and(lower, right)
    UL = np.logical_and(upper, left)
    UR = np.logical_and(upper, right)
    
    LL_ = np.count_nonzero(LL)
    LR_ = np.count_nonzero(LR)
    UL_ = np.count_nonzero(UL)
    UR_ = np.count_nonzero(UR)
    return ([LL_, LR_, UL_, UR_], [LL, LR, UL, UR])

def arrayIndexFromPos(pos, density):
    """Returns the array index corresponding to pos as a tuple.
    pos should be a numpy array."""
    return tuple(np.array(density * pos, dtype=np.int0))
