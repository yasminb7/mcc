'''
Created on Jul 18, 2012

@author: frederic
'''
import os, datetime
import scipy as sp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from classMesenchymal import Mesenchymal
from classMaze import Maze, densityArray
from classDataset import Dataset
from gradient import concentration
import statutils
import sim
import utils
from scipy import random
from utils import info
import constants

class Simulation:
    
    def __init__(self, const, noDelete=False):
        self.const = const
        
        self.path = os.getcwd() + "/"
        self.resultsdir = os.path.join(self.path, constants.resultspath, const["name"])
        
        if not os.path.exists(self.resultsdir):
            os.mkdir(self.resultsdir)
        if noDelete==False:
            utils.makeFolderEmpty(self.resultsdir)
            
        self.N_amoeboid = self.const["N_amoeboid"]
        self.N_mesenchymal = self.const["N_mesenchymal"]
        self.N = self.N_amoeboid + self.N_mesenchymal
        self.NNN = int(const["max_time"] / const["dt"])
        
        if noDelete==False:
            self.dsA = Dataset(Dataset.AMOEBOID, const["max_time"], const["dt"], self.N, constants.DIM, self.resultsdir, fileprefix=None)
            info("Created dataset of size %s" % self.dsA.getHumanReadableSize())
    
    def plotmaze(self, maze, filename = "maze.png"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(maze.T, extent=self.const["fieldlimits"], cmap=cm.get_cmap("binary"))
        plt.grid(True)
        plt.axis(self.const["fieldlimits"])
        mazefilepath = utils.getResultsFilepath(self.resultsdir, filename)
        plt.savefig(mazefilepath)
        
    def plotcgrad(self, concentrationfield, filename = 'concentrationfield.png'):
        fig = plt.figure(figsize=(16,16), frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.], )
        ax.set_axis_off()
        fig.add_axes(ax)
        img = utils.scaleToMax(1.0, -concentrationfield.T)
        ax.imshow(img, extent=self.const["fieldlimits"], origin='lower')
        concentrationfilepath = utils.getResultsFilepath(self.resultsdir, filename)
        plt.savefig(concentrationfilepath, bbox_inches='tight', dpi=100)
    
    #@profile
    def run(self):
        logfilepath = utils.getResultsFilepath(self.resultsdir, "logfile.txt")
        logging_handler = utils.setup_logging_sim(logfilepath)
            
        t_start = datetime.datetime.now()
        DIM = constants.DIM
        N = self.N
        N_mesenchymal = self.N_mesenchymal
        N_amoeboid = self.N_amoeboid
        NNN = self.NNN
        dt = self.const["dt"]
        goal = sp.array(self.const["gradientcenter"])
        initial_position = sp.array(self.const["initial_position"])
        stray = sp.array(self.const["initial_position_stray"])
        assert stray>0, "Unfortunately you have to use a stray>0."
        
        info("Simulation started with %s agents" % N)
        
        #prepare landscape
        mazepath = os.path.join(self.path, self.const["maze"])
        myMaze = Maze(mazepath, self.const["fieldlimits"], self.const["border"])
        maze = myMaze.getData()
        density = densityArray(maze.shape, self.const["fieldlimits"])
        #prepare gradient
        goal_px = sp.array(density * goal, dtype=sp.int_)
        concentrationfield = sp.fromfunction(lambda x, y: concentration(x, y, goal_px), (maze.shape))        
        #or null field
        #concentrationfield = sp.ones((x_max, y_max))
        cgrad = utils.getGradient(concentrationfield)
        
        #initialize agents
        self.dsA.types = sp.array(N_mesenchymal*[Dataset.is_mesenchymal]+N_amoeboid*[Dataset.is_amoeboid])
        isMesenchymal = self.dsA.types==Dataset.is_mesenchymal
        isAmoeboid = self.dsA.types==Dataset.is_amoeboid
        self.dsA.velocities[0] = sp.zeros((N, DIM))
        self.dsA.energies[0] = 0.0
        self.dsA.states[0] = sim.States.MOVING
        period = self.const["orientationperiod"] + sp.random.standard_normal(N) * self.const["periodsigma"]
        delay = self.const["orientationdelay"] + sp.random.standard_normal(N) * self.const["delaysigma"]
        self.dsA.periods = period
        self.dsA.delays = delay
        self.dsA.eating[0] = False
        self.dsA.times[0] = 0.0
        self.dsA.statechanges = period * sp.random.random(N)
        
        Mesenchymal.classInit(self.const)
        for agentIndex in range(N_mesenchymal+N_amoeboid):
            tryToPlace = True
            nAttempts = 0
            while tryToPlace:
                initX = sp.random.normal(initial_position[0], stray)
                initY = sp.random.normal(initial_position[1], stray)
                posidx = sp.array(density * sp.array([initX, initY]), dtype=sp.int0)
                sp.clip(posidx, myMaze.minidx, myMaze.maxidx, out=posidx)
                onECM = (maze[posidx[0],posidx[1]] > 0.5 * constants.wallconst)
                if not onECM:
                    self.dsA.positions[0,agentIndex] = sp.array([initX, initY])
                    tryToPlace = False
                else:
                    nAttempts += 1
                    assert nAttempts < 15, "Could not place agents outside of ECM for given parameters"
        
        posidx = sp.array(density * self.dsA.positions[0], dtype=sp.int0)
        
        x = sp.array(cgrad[0,posidx[:,0],posidx[:,1]])
        y = sp.array(cgrad[1,posidx[:,0],posidx[:,1]])
        self.dsA.direction_angles = sp.arctan2(y, x)
        self.dsA.direction_angles[isMesenchymal] += self.const["compass_noise_m"] * sp.random.standard_normal(N_mesenchymal)
        self.dsA.direction_angles[isAmoeboid] += self.const["compass_noise_a"] * sp.random.standard_normal(N_amoeboid)
        
        if isMesenchymal.any():
            assert Mesenchymal.getMaxEatingDistance(density)<=myMaze.border, "The area where Mesenchymal eat is too large compared to the border of the maze"
        
        #how often do we want to update the user about the progress (period)
        update_progress = 50
        do_continue = True
        max_time = self.const["max_time"]
        lasttime = dt
        simtime = dt
        margin = self.const["border"]
        fieldlimits = self.const["fieldlimits"]
        iii = 1
        enable_interaction = self.const["enable_interaction"]
        notDiag = ~sp.eye(N, dtype=sp.bool_)
        r_coupling = self.const["repulsion_coupling"]
        w = self.const["w"] * dt
        _w = (1-w) * dt
        interaction_radius = self.const["interaction_radius"]
        alignment_radius = self.const["alignment_radius"]
        q = self.const["q"]
        eta = self.const["eta"]
        gamma_a = self.const["gamma_a"]
        gamma_m = self.const["gamma_m"]
        r = self.const["r"]
        one_over_m = 1/self.const["mass"]
        success_radius = self.const["success_radius"]
        delta = self.const["delta"]
        compass_noise_a = self.const["compass_noise_a"]
        compass_noise_m = self.const["compass_noise_m"]
        directions = sp.zeros((N, DIM))
        directions[:,0] = sp.cos(self.dsA.direction_angles)
        directions[:,1] = sp.sin(self.dsA.direction_angles)
        degradation = sp.zeros(N)
        degradation_radius = self.const["degradation_radius"]
        nodegradationlimit = self.const["nodegradationlimit"]
        
        positions = self.dsA.positions
        velocities = self.dsA.velocities
        energies = self.dsA.energies
        states = self.dsA.states
        statechanges = self.dsA.statechanges
        eating = self.dsA.eating
        periods = self.dsA.periods
        delays = self.dsA.delays
        times = self.dsA.times
        direction_angles = self.dsA.direction_angles
        has_had_update = sp.zeros(direction_angles.shape, dtype=sp.int_)
        
        dragFactor = sp.empty((N,))
        F_repulsion = sp.zeros((N,2))

        while do_continue:
            if simtime-lasttime>update_progress:
                info("Time %0.2f out of maximum %s" % (simtime,max_time))
                lasttime = simtime
            
            states[iii] = states[iii-1]     
            E = energies[iii-1]
            times[iii] = simtime

            F_repulsion[:] = 0.0
            if enable_interaction:
                #calculate distances between agents
                #we create an array containing in N times the array containing the agents' positions
                #and we substract from that array its transpose (of some sort)
                pos_v = sp.array(N*[positions[iii-1]])
                pos_h = pos_v.swapaxes(0,1)
                pos_difference = pos_v-pos_h
                d_squared = sp.sum(pos_difference*pos_difference, axis=2)
                
                inRepulsionRadius = d_squared < 4*interaction_radius*interaction_radius
                inAlignmentRadius = d_squared < 4*alignment_radius*alignment_radius
                doRepulse = sp.logical_and(inRepulsionRadius, notDiag)
                doAlign = sp.logical_and(~inRepulsionRadius, inAlignmentRadius)
                doAlign = sp.logical_and(doAlign, notDiag)
                dir_x = directions[:,0]
                dir_y = directions[:,1]
                newdir_x = dir_x
                newdir_y = dir_y
                for i in xrange(N):
                    interactionPartners = doAlign[i]
                    nInteractions = sp.count_nonzero(interactionPartners)
                    if nInteractions>0:
                        newdir_x[i] = _w*dir_x[i] + w*sp.mean( dir_x[interactionPartners] )
                        newdir_y[i] = _w*dir_y[i] + w*sp.mean( dir_y[interactionPartners] )
                dir_norms = sp.sqrt( newdir_x*newdir_x + newdir_y*newdir_y )
                dir_norms[dir_norms==0.0] = 1.0
                directions[:,0] = newdir_x/dir_norms
                directions[:,1] = newdir_y/dir_norms

                F_rep = sp.zeros_like(pos_difference)
                interactionIndices = sp.nonzero(doRepulse)
                for i, j in zip(interactionIndices[0], interactionIndices[1]):
                    F_rep[i,j] = r_coupling * pos_difference[i,j,:] / d_squared[i,j,sp.newaxis]
                F_repulsion = sp.sum(F_rep, axis=0)

            posidx = sp.array(density * positions[iii-1], dtype=sp.int0)
            sp.clip(posidx, myMaze.minidx, myMaze.maxidx, out=posidx)
            wgrad = myMaze.getGradientsPython(posidx)

            evolve_orientation = statechanges < simtime 
            is_moving = states[iii-1]==sim.States.MOVING
            is_orienting = states[iii-1]==sim.States.ORIENTING
            theta = sp.zeros(self.N)
            theta[is_moving] = 1.0
            
            switch_to_orienting = sp.logical_and(evolve_orientation, is_moving)
            if switch_to_orienting.any():
                states[iii][switch_to_orienting] = sim.States.ORIENTING
                statechanges[switch_to_orienting] = simtime + delays[switch_to_orienting]

            switch_to_moving = sp.logical_and(evolve_orientation, is_orienting)
            if switch_to_moving.any():
                states[iii][switch_to_moving] = sim.States.MOVING
                statechanges[switch_to_moving] = simtime + periods[switch_to_moving]
                x = sp.array(cgrad[0,posidx[switch_to_moving,0],posidx[switch_to_moving,1]])
                y = sp.array(cgrad[1,posidx[switch_to_moving,0],posidx[switch_to_moving,1]])
                direction_angles[switch_to_moving] = sp.arctan2(y, x)
                switch_to_moving_a = sp.logical_and(switch_to_moving, isAmoeboid)
                switch_to_moving_m = sp.logical_and(switch_to_moving, isMesenchymal)
                direction_angles[switch_to_moving_a] += compass_noise_a * sp.random.standard_normal(sp.count_nonzero(switch_to_moving_a))
                direction_angles[switch_to_moving_m] += compass_noise_m * sp.random.standard_normal(sp.count_nonzero(switch_to_moving_m))
                
                directions[switch_to_moving, 0] = sp.cos(direction_angles[switch_to_moving])
                directions[switch_to_moving, 1] = sp.sin(direction_angles[switch_to_moving])
            
            propelFactor = eta * E * theta
            F_propulsion = propelFactor[:,sp.newaxis] * directions
             
            dragFactor[isAmoeboid] = - gamma_a
            dragFactor[isMesenchymal] = - gamma_m
            
            F_drag = dragFactor[:,sp.newaxis] * velocities[iii-1]
            
            F_stoch = r * random.standard_normal(size=F_propulsion.shape)
            
            F_total = F_propulsion + F_drag + F_stoch + F_repulsion
            
            #Check if new position is allowed, otherwise move back 
            ta = positions[iii-1,:,0]<fieldlimits[0]+margin
            tb = positions[iii-1,:,0]>fieldlimits[1]-margin
            tc = positions[iii-1,:,1]<fieldlimits[2]+margin
            td = positions[iii-1,:,1]>fieldlimits[3]-margin
            outsidePlayingField = sp.logical_or(ta, tb)
            outsidePlayingField = sp.logical_or(outsidePlayingField, tc)
            outsidePlayingField = sp.logical_or(outsidePlayingField, td)
            
            nda = positions[iii-1,:,0]<fieldlimits[0]+nodegradationlimit
            ndb = positions[iii-1,:,0]>fieldlimits[1]-nodegradationlimit
            ndc = positions[iii-1,:,1]<fieldlimits[2]+nodegradationlimit
            ndd = positions[iii-1,:,1]>fieldlimits[3]-nodegradationlimit
            noDegradation = sp.logical_or(nda, ndb)
            noDegradation = sp.logical_or(noDegradation, ndc)
            noDegradation = sp.logical_or(noDegradation, ndd)            
            
            wallnorm = sp.sum(wgrad * wgrad, axis=1)
            touchingECM = (wallnorm != 0.0)

            collidingWithWall = sp.logical_and(touchingECM, isAmoeboid)
            collidingWithWall = sp.logical_or(collidingWithWall, outsidePlayingField)
            collidingWithWall = sp.logical_or(collidingWithWall, sp.logical_and(isMesenchymal, noDegradation))

            if collidingWithWall.any():
                dotF = sp.zeros((self.N))
                dotF[collidingWithWall] = sp.sum(F_total[collidingWithWall] * wgrad[collidingWithWall], axis=1)
                doAdjust = dotF<0
                if doAdjust.any():
                    dotF[doAdjust] /= wallnorm[doAdjust]
                    F_total[doAdjust] -=  dotF[doAdjust][:,sp.newaxis] * wgrad[doAdjust]

            velocities[iii] = velocities[iii-1] + one_over_m * dt * F_total
            
            if collidingWithWall.any():
                dotv = sp.zeros((self.N))
                dotv[collidingWithWall] = sp.sum(velocities[iii][collidingWithWall] * wgrad[collidingWithWall], axis=1)
                doAdjust = dotv<0
                if doAdjust.any():
                    dotv[doAdjust] /= wallnorm[doAdjust]
                    velocities[iii][doAdjust] -= dotv[doAdjust][:,sp.newaxis] * wgrad[doAdjust]

            positions[iii] = positions[iii-1] + velocities[iii] * dt
            
            #TODO: next line don't let the ones outside the playing field degrade
            posidx = sp.array(density * positions[iii-1], dtype=sp.int0)
            sp.clip(posidx, myMaze.minidx, myMaze.maxidx, out=posidx)
            ECM = maze[posidx[:,0],posidx[:,1]]
            #TODO remove hardcoded value here
            inOriginalECM = ECM > 0.7 * constants.wallconst
            inContactWithECM = sp.logical_or(touchingECM, inOriginalECM)
            isDegrading = sp.logical_and(isMesenchymal, inContactWithECM)
            isDegrading = sp.logical_and(isDegrading, ~noDegradation)

            #if an agent is degrading now then he should be degrading during the next 'degrading_bias' steps
            vel = velocities[iii][:]
            v = sp.sqrt(sp.sum(vel*vel, axis=1))

            for i in isDegrading.nonzero()[0]:
                #degradation_radius has to be in distance units, not pixels!
                bias = int(degradation_radius/(v[i]*dt * density[0] )) if v[i] > 0 else 0
                nextIdx = min(NNN, iii+bias)
                eating[iii:nextIdx, i] = True
            
            kappa = sp.logical_or(isDegrading, eating[iii])
            
            for pos, energy in zip(positions[iii][eating[iii]], E[eating[iii]]):
                myMaze.degrade(pos, Mesenchymal.eat, energy, density, dt)
            
            needs_update = isDegrading
            has_had_update[needs_update] = iii

            degradation[:] = 0.0
            degradation[kappa] = Mesenchymal.degrad_rate
            
            energies[iii] = E + (q - delta * E - theta * eta * v * E - degradation * E) * dt
            energies[iii][ energies[iii]<0.0 ] = 0.0
            
            distances = statutils.getDistances(positions[iii], goal)
            states[iii][distances < success_radius] = sim.States.CLOSE
            
            simtime += dt
            iii += 1
            do_continue = iii<NNN

        self.iii = iii-1
        self.simtime = simtime
        t_stop = datetime.datetime.now()
        info( "Simulation ended ")
        info( "Simulation took %s and ended with a simulation time of %s" % (t_stop-t_start, simtime))
        
        if self.dsA is not None:
            self.dsA.resizeTo(iii)
            self.dsA.saveTo(self.resultsdir)
        
        #save the maze of the last time step so we can reuse it for the path plots
        with open(os.path.join(self.resultsdir, constants.finalmaze_filename), "w") as finalmaze:
            sp.save(finalmaze, myMaze.data)
        
        utils.savetimestamp(self.resultsdir)
        utils.remove_logging_sim(logging_handler)
    
    def saveFinalGradient(self, myMaze):
        from PIL import Image
        imgdata = myMaze.getGradImageSquared()
        imgdata = utils.scaleToMax(255.0, imgdata.T)
        imgdata = sp.flipud(imgdata)
        img = Image.fromarray(imgdata.astype('uint8'))
        img.save(os.path.join(self.resultsdir, "lastGradient.bmp"))
    
    def reportLastMazeGrad(self, steps, iii, idxA, positions, density, mazegrad, times, number):
        import plotting
        bk = steps
        line = positions[iii-bk:iii, idxA]
        posidx = sp.array(density * line, dtype=sp.int_)
        mazepoints = mazegrad[:,posidx[:,0],posidx[:,1]]
        mazepoints = sp.sqrt(mazepoints[0]*mazepoints[0] + mazepoints[1]*mazepoints[1])
        y = [mazepoints]
        plotting.plot([range(len(mazepoints))] , y, folder=self.resultsdir, savefile="mazeline_%d.png" % number)
    
    def analyse(self):
        picklepath = utils.getResultsFilepath(self.resultsdir, constants.finalstats_pickle)
        textpath = utils.getResultsFilepath(self.resultsdir, constants.finalstats_text)
        statutils.savefinalstats(picklepath, textpath, self.const, self.dsA)

def prepareSim(const):
    utils.setup_logging_base() 
    #run the simulation
    mySim = Simulation(const)
    mySim.run()
    #run analysis algorithms
    mySim.analyse()
    functions = []
    import graphics
    if const["create_video_directly"]:
        functions.append(graphics.create_plots_ds)
    if const["create_path_plot"]:
        functions.append(graphics.create_path_plot)
    graphics.writeFrames(const, output_func=functions)
    
if __name__ == "__main__":
    simfile = constants.currentsim
    #TODO: should add unravel here
    const = utils.readConst(constants.simdir, simfile)
    const = const["get"](tuple())
    prepareSim(const)