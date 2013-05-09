'''
Created on 14.10.2012

@author: frederic
'''
#TODO: remove const from arguments of functions that deal with pickles only

import os, csv
import utils, statutils, plotting
import scipy as sp
import constants
import random
import classDataset
from classDataset import Dataset
from constants import simdir, resultspath
import sim
import analytics
import itertools

def prepareStats(filename):
    """Return const and savefolder for a given simfile."""
    cwd = os.getcwd()
    savefolder = filename[0:-3]
    const = utils.readConst(simdir, filename)
    savefolder = os.path.join(cwd, resultspath, savefolder)
    dummyfile = os.path.join(savefolder, "dummy.txt")
    utils.ensure_dir(dummyfile)
    return (const, savefolder)

def remove_None(y_val, legend):
    assert len(y_val)==len(legend)
    new_y = []
    new_l = []
    for i, el in enumerate(y_val):
        if el is not None:
            new_y.append(el)
            new_l.append(legend[i])
    return (new_y, new_l) if len(new_y)>0 else (y_val, legend)

def singleStats(filename, doFinalStats=False):
    """Create some statistics for a given simulation and save them in the same directory.
    Statistics include for example the energy and velocity plots for a few agents,
    the same plots averaged over all agents, path length distributions and chemotactic indices.
    """
    const, savefolder = prepareStats(filename)
    root = os.getcwd()
    utils.ensure_dir(savefolder)
    
    unraveled = utils.unravel(const)
    #If you have a sim-file causing a large number of simulations, but you want to
    #create statistics for only a few of them, you can use utils.applyFilter() as in this example:
    #unraveled = utils.applyFilter(unraveled, "repetitions", [0])
    
    for results in unraveled:
        print "singleStats for %s" % results["name"]
        _resultspath = constants.resultspath[0:-1]
        resultsfolder = os.path.join(root, _resultspath, results["name"])
        ds = classDataset.load(Dataset.AMOEBOID, resultsfolder, fileprefix="A", dt=results["dt"])
        statutils.clearStatistics(resultsfolder)
        
        if doFinalStats:
            picklepath = utils.getResultsFilepath(resultsfolder, constants.finalstats_pickle)
            textpath = utils.getResultsFilepath(resultsfolder, constants.finalstats_text)
            statutils.savefinalstats(picklepath, textpath, results, ds)
        mesenchymals = (ds.types==ds.is_mesenchymal)
        amoeboids = (ds.types==ds.is_amoeboid)
        idx_m = sp.where(mesenchymals)[0]
        idx_a = sp.where(amoeboids)[0]
        sample_m = random.sample(idx_m, min(2, len(idx_m))) if len(idx_m)>5 else list(idx_m)
        sample_a = random.sample(idx_a, min(2, len(idx_a))) if len(idx_a)>5 else list(idx_a)
        sampled_agents = sample_m + sample_a 
        for i in sampled_agents:
            N = -1
            myylimits = None
            vel = ds.velocities[:,i]
            v = sp.sqrt(sp.sum(vel*vel, axis=1))
            typechar = 'm' if ds.types[i]==ds.is_mesenchymal else 'a' 
            plotting.plotlines_vert_subplot([ds.times[:N]], [ds.energies[:N,i]], [ds.times[:N]], [v[:N]], xlabel="Time", ylabel="Energy", xlabel2="Time", ylabel2="Velocity", ylimits=myylimits, folder=resultsfolder, savefile="energy_%s%s%s" % (i, typechar, constants.graphics_ending))

        #get average energy
        consider = sp.logical_or(ds.states==sim.States.MOVING, ds.states==sim.States.ORIENTING)
        energies = sp.ma.array(ds.energies, mask=~consider)
        avg_energy_m = sp.mean(energies[:,mesenchymals], axis=1) if mesenchymals.any() else None
        avg_energy_a = sp.mean(energies[:,amoeboids], axis=1) if amoeboids.any() else None
        #get avg velocities
        vel = ds.velocities
        v = sp.sqrt(sp.sum(vel*vel, axis=2))
        v_masked = sp.ma.array(v, mask=~consider)
        avg_vel_m = sp.mean(v_masked[:,mesenchymals], axis=1) if mesenchymals.any else None
        avg_vel_a = sp.mean(v_masked[:,amoeboids], axis=1) if amoeboids.any() else None
        #plot the stuff on two separate axes
        y_axes = [avg_energy_a, avg_energy_m]
        y_axes2 = [avg_vel_a, avg_vel_m]
        mylegend = ["amoeboid", "mesenchymal"]
        mylegend2 = mylegend
        y_axes, mylegend = remove_None(y_axes, mylegend)
        y_axes2, mylegend2 = remove_None(y_axes2, mylegend2)
        plotting.plotlines_vert_subplot(len(y_axes)*[ds.times], y_axes, len(y_axes2)*[ds.times], y_axes2, xlabel="Time", ylabel="Average energy", xlabel2="Time", ylabel2="Average velocity", legend=mylegend,legend2=mylegend2, folder=resultsfolder, savefile=constants.avg_en_vel_filename)
        
        goal = sp.array(results["gradientcenter"])
        dist = statutils.getDistances(ds.positions, goal)
        dist_m = dist[:,mesenchymals] 
        dist_a = dist[:,amoeboids] 
        #dist_m = statutils.getDistances(ds.positions[:,mesenchymals], goal) if mesenchymals.any()==True else None
        #dist_a = statutils.getDistances(ds.positions[:,amoeboids], goal) if amoeboids.any()==True else None
#        avg_distance_m = sp.mean( dist_m, axis=1) if mesenchymals.any() else None
#        avg_distance_a = sp.mean( dist_a, axis=1) if amoeboids.any() else None
        
        success_radius = results["success_radius"]
        successful = dist[-1] <= success_radius
#        success_rate_m = statutils.getSuccessrates(dist_m, success_radius) if dist_m is not None else None
#        success_rate_a = statutils.getSuccessrates(dist_a, success_radius) if dist_a is not None else None
        
        mylegend = ["amoeboid", "mesenchymal"]
        
#        y_axes_dist = [avg_distance_a, avg_distance_m]
#        y_axes_dist, mylegend_dist = remove_None(y_axes_dist, mylegend) 
#        plotting.plotlines(len(y_axes_dist)*[ds.times], y_axes_dist, xlabel="Time", ylabel="Distance from goal", legend=mylegend_dist, folder=resultsfolder, savefile=constants.avg_dist_filename)
        
#        y_axes_succ = [success_rate_a, success_rate_m]
#        y_axes_succ, mylegend_succ = remove_None(y_axes_succ, mylegend)
#        plotting.plotlines(len(y_axes_succ)*[ds.times], y_axes_succ, xlabel="Time", ylabel="Success rate", legend=mylegend_succ, folder=resultsfolder, savefile=constants.succ_rate_filename)
        
        #create path length distribution
        pathlengths = sp.sum(v_masked*results["dt"], axis=0)
        plotting.plot_histogram(pathlengths, nbins=20, xlabel="Path length", title="Path length distribution", folder=resultsfolder, savefile=constants.pathlengths_filename)
        if mesenchymals.any():
            plotting.plot_histogram(pathlengths[mesenchymals], nbins=20, xlabel="Path length", title="Path length distribution", folder=resultsfolder, savefile=constants.pathlengths_filename_m)
        if amoeboids.any():
            plotting.plot_histogram(pathlengths[amoeboids], nbins=20, xlabel="Path length", title="Path length distribution", folder=resultsfolder, savefile=constants.pathlengths_filename_a)
        
        #calculate chemotactic index
        displacement = ds.positions[-1] - ds.positions[0]
        #displacement_n = sp.sqrt( sp.sum(displacement*displacement, axis=1))
        #displacement_u = displacement / displacement_n
        straightPath = results["gradientcenter"]-ds.positions[0]
        straightPath_n = sp.sqrt( sp.sum(straightPath*straightPath, axis=1))
        straightPath_u = sp.zeros_like(straightPath)
        for i in sp.nonzero(straightPath_n):
            straightPath_u[i] = straightPath[i] / straightPath_n[i][:,sp.newaxis]
        #minPathLengths = sp.sqrt( sp.sum( straightPath**2 , axis=1))
        #ci = minPathLengths/pathlengths
        #ci = sp.fromiter( [sp.sum( str*disp ) for str, disp in zip(straightPath_u, displacement)], dtype=sp.float64)
        ci = sp.sum(straightPath_u*displacement/pathlengths[:,sp.newaxis], axis=1)
        plotting.plot_histogram(ci, nbins=20, xlabel="Chemotactic index", title="CI distribution", folder=resultsfolder, savefile=constants.ci_filename, xlim=(0,None))
        
        if mesenchymals.any():
            plotting.plot_histogram(ci[mesenchymals], nbins=20, xlabel="Chemotactic index", title="CI distribution (M)", folder=resultsfolder, savefile=constants.ci_filename_m, xlim=(0,None))
        if amoeboids.any():
            plotting.plot_histogram(ci[amoeboids], nbins=20, xlabel="Chemotactic index", title="CI distribution (A)", folder=resultsfolder, savefile=constants.ci_filename_a, xlim=(0,None))
        
        #plot stacked path length histograms
        Nbins = 20
        hist, edges = sp.histogram(pathlengths, bins=Nbins)
        histp, edgesp = sp.histogram(pathlengths[successful==True], bins=Nbins, range=(edges[0], edges[-1]))
        histn, edgesn = sp.histogram(pathlengths[successful==False], bins=Nbins, range=(edges[0], edges[-1]))
        left = []
        width = []
#        for i in xrange(hist.size):
#            left.append(edges[i])
#            width.append(edges[i+1]-edges[i])
#        plotting.bars_stacked(histp, 'b', histn, 'g', left, width, folder=resultsfolder, savefile=constants.pathlengths_stacked)
        
        #scatterplot of average velocities of each agent
#        scatter_vel = sp.mean( ds.velocities, axis=0 )
#        plotting.scatter(scatter_vel[:,0], scatter_vel[:,1], amoeboids, mesenchymals, successful, folder=resultsfolder, savefile=constants.scattervel_filename)
        
        plotting.scattercisr(ci, successful, amoeboids, mesenchymals, xlabel="CI", ylabel="success", folder=resultsfolder, savefile=constants.cisr_filename)
                
        speedDegrading = analytics.equilSpeedDegrading(results, results["q"])
        print "Speed while degrading %s" % (speedDegrading,)
        speedPropulsing = analytics.equilSpeedPropulsing(results, results["q"])
        print "Speed while propulsing %s" % (speedPropulsing,)
        print "Ratio between the two %s" % (speedPropulsing/speedDegrading,)
        

def plotGlobal(xaxis, lines, const, statvar, savedir, filename, disableVarInLegend=False, **plotargs):
    print "Plotting %s" % (statvar,)
    mylines = []
    possiblePlot = xaxis in const["factors"]
    if not possiblePlot:
        print "Cannot create this kind of plot given the arguments %s and %s" % (xaxis, lines)
        return None
    
    unraveled = utils.unravel(const)
    
    if lines in const["factors"]:
        linesiter = const[lines]
    else:
        linesiter = [const[lines]]
    
    for line in linesiter:
        linedata = {"xs" : const[xaxis]}
        ys = []
        yerrs = []
        for x in const[xaxis]:
            myconst = utils.applyFilter(unraveled, xaxis, [x])
            myconst = utils.applyFilter(myconst, lines, [line])
            value, err = getFinalstats(myconst, statvar)
            ys.append(value)
            yerrs.append(err)
        linedata["ys"] = ys
        linedata["yerrs"] = yerrs
        linedata["label"] = "%s = %s" % (lines, constants.symbol(line)) if disableVarInLegend==False else constants.symbol(line)
        mylines.append(linedata)
    
    myxaxes = [line["xs"] for line in mylines]
    myyaxes = [line["ys"] for line in mylines]
    myerrors = [line["yerrs"] for line in mylines]
    mylegend = [line["label"] for line in mylines]
    myxlabel = constants.symbol(xaxis)
    myylabel = constants.symbol(statvar)
    
#    if statvar=="avg_en":
#        import analytics as an
#        an.estimateAvgEnergy(const, ECMDensity, N_a, N_m)
    if statvar=="success_ratio":
        with open(os.path.join(savedir, 'success_rates.csv'), 'wb') as _file:
            csvfile = csv.writer(_file, dialect=csv.excel)
            for line in mylines:
                csvfile.writerow([line["label"]])
                csvfile.writerow(line["xs"])
                csvfile.writerow(line["ys"])
                csvfile.writerow(line["yerrs"])
        
    plotting.errorbars(myxaxes, myyaxes, y_bars=myerrors, legend=mylegend, xlabel=myxlabel, ylabel=myylabel, folder=savedir, savefile=filename, **plotargs)

def plotGlobal2factors(xaxis, lines, const, statvar, savedir, filename, disableVarInLegend=False, **plotargs):
    """Same as plotGlobal, but here lines is a list. We will take the cartesion product
    of the factors in the list and plot a line for each combination.
    """
    #If I can't access lines as a list, there's nothing to do.
    assert not isinstance(lines, basestring), "Cannot create this kind of plot: lines is not a list."
    
    print "Plotting %s" % (statvar,)
    mylines = []
    possiblePlot = xaxis in const["factors"]
    
    allInFactors = True
    for l in lines:
        if l not in const["factors"]:
            allInFactors = False
            break
        
    if not possiblePlot or not allInFactors:
        print "Cannot create this kind of plot given the arguments %s and %s" % (xaxis, lines)
        return None
    
    unraveled = utils.unravel(const)
    
    linesiter = itertools.product( const[lines[0]], const[lines[1]]) 
    
    for line in linesiter:
        linedata = {"xs" : const[xaxis]}
        ys = []
        yerrs = []
        for x in const[xaxis]:
            myconst = utils.applyFilter(unraveled, xaxis, [x])
            myconst = utils.applyFilter(myconst, lines[0], [line[0]])
            myconst = utils.applyFilter(myconst, lines[1], [line[1]])
            value, err = getFinalstats(myconst, statvar)
            ys.append(value)
            yerrs.append(err)
        linedata["ys"] = ys
        linedata["yerrs"] = yerrs
        interaction = line[1]
        if interaction==True:
            interaction='+'
        elif interaction==False:
            interaction='-'
        label = "q = %s, %s" % (line[0], interaction)
        linedata["label"] = label if disableVarInLegend==False else constants.symbol(line)
        mylines.append(linedata)
    
    myxaxes = [line["xs"] for line in mylines]
    myyaxes = [line["ys"] for line in mylines]
    myerrors = [line["yerrs"] for line in mylines]
    mylegend = [line["label"] for line in mylines]
    myxlabel = constants.symbol(xaxis)
    myylabel = constants.symbol(statvar)
        
    plotting.errorbars(myxaxes, myyaxes, y_bars=myerrors, legend=mylegend, xlabel=myxlabel, ylabel=myylabel, folder=savedir, savefile=filename, **plotargs)
    
def plotFitness(xaxis, lines, const, savedir, filename, suffix=None):
    mylines = []
    assert xaxis in const["factors"] and lines in const["factors"]
    
    unraveled = utils.unravel(const)
    
    for line in const[lines]:
        linedata = {"xs" : const[xaxis]}
        ys = []
        for x in const[xaxis]:
            myconst = utils.applyFilter(unraveled, xaxis, [x])
            myconst = utils.applyFilter(myconst, lines, [line])
            success_ratio, sr_err = getFinalstats(myconst, "success_ratio") if suffix is None else getFinalstats(myconst, "success_ratio_%s" % suffix) 
            avg_en, ae_err = getFinalstats(myconst, "avg_en") if suffix is None else getFinalstats(myconst, "avg_en_%s" % suffix)
            value = success_ratio * avg_en
            ys.append(value)
        linedata["ys"] = ys
        linedata["label"] = "%s = %s" % (lines, line)
        mylines.append(linedata)
    
    myxaxes = [line["xs"] for line in mylines]
    myyaxes = [line["ys"] for line in mylines]
    mylegend = [line["label"] for line in mylines]
    myxlabel = constants.symbol(xaxis)
    myylabel = constants.symbol("fitness")
    
    plotting.errorbars(myxaxes, myyaxes, legend=mylegend, xlabel=myxlabel, ylabel=myylabel, folder=savedir, savefile=filename)
    
def plotFitness2(xaxis, lines, const, savedir, filename, suffix=None):
    """Success ratio times avg energy of those who where successful"""
    mylines = []
    assert xaxis in const["factors"] and lines in const["factors"]
    
    unraveled = utils.unravel(const)
    
    for line in const[lines]:
        linedata = {"xs" : const[xaxis]}
        ys = []
        for x in const[xaxis]:
            myconst = utils.applyFilter(unraveled, xaxis, [x])
            myconst = utils.applyFilter(myconst, lines, [line])
            values = []
            for c in myconst:
                datapath = os.path.join(constants.resultspath, c["name"])
                ds = classDataset.load(Dataset.AMOEBOID, datapath, dt=c["dt"], fileprefix="A")
                dist = statutils.getDistances(ds.positions, c["gradientcenter"])
                amoeboids = ds.types==Dataset.is_amoeboid
                mesenchymal = ds.types==Dataset.is_mesenchymal
                successful = dist[-1] < c["success_radius"]
                consider = sp.logical_or(ds.states==sim.States.MOVING, ds.states==sim.States.ORIENTING)
                energies = sp.ma.masked_array(ds.energies, ~consider)
                success_ratio = sp.mean(successful)
                avg_en = sp.mean(energies[:,successful])
                #success_ratio = getFinalstats(myconst, "success_ratio") if suffix is None else getFinalstats(myconst, "success_ratio_%s" % suffix) 
                #avg_en = getFinalstats(myconst, "avg_en") if suffix is None else getFinalstats(myconst, "avg_en_%s" % suffix)
                values.append(success_ratio * avg_en)
            value = None
            if c["handle_repetitions_with"]=="mean":
                value = sp.mean(values)
            elif c["handle_repetitions_with"]=="median":
                value = sp.median(values)
            ys.append(value)
        linedata["ys"] = ys
        linedata["label"] = "%s = %s" % (lines, line)
        mylines.append(linedata)
    
    myxaxes = [line["xs"] for line in mylines]
    myyaxes = [line["ys"] for line in mylines]
    mylegend = [line["label"] for line in mylines]
    myxlabel = constants.symbol(xaxis)
    myylabel = constants.symbol("fitness")
    
    plotting.errorbars(myxaxes, myyaxes, legend=mylegend, xlabel=myxlabel, ylabel=myylabel, folder=savedir, savefile=filename)

def getFinalstats(constlist, statvar):
    statistics = sp.zeros(len(constlist))
    for i, c in enumerate(constlist):
        f = c["name"]
        finalstatspath = os.path.join(resultspath, f, constants.finalstats_pickle)
        finalstats = statutils.readfinalstats(finalstatspath)
        statistics[i] = finalstats[statvar]
    value = None
    error = None
    if c["handle_repetitions_with"]=="mean":
        value = sp.mean(statistics)
        error = sp.std(statistics)
    elif c["handle_repetitions_with"]=="median":
        value = sp.median(statistics)
        error = sp.std(statistics)
    return value, error

def plotCombinations(xaxis, lines, const, statvar, savedir, filename, suffixes=None, **plotargs):
    if suffixes is None:
        #suffixes = ["", "_m", "_a", "_m_s", "_m_us", "_a_s", "_a_us"]
        suffixes = ["", "_m", "_a"]
    for suff in suffixes:
        plotGlobal(xaxis, lines, const, statvar+suff, savedir, filename+suff+constants.graphics_ending, **plotargs)
  
if __name__=="__main__":
    simfile = constants.currentsim
    print "Creating statistics for %s" % simfile
    
    #create plots for individual simulations
    #singleStats(simfile, doFinalStats=False)
    
    #create plots containing data from the entire set of simulations described by currentsim
    cwd = os.getcwd()
    simdir = os.path.join(cwd, constants.simdir)
    const = utils.readConst(simdir, simfile)
    savedir = utils.getResultsDir(simfile)
    savedir = os.path.join(cwd, constants.resultspath, savedir)
    utils.ensure_dir(savedir)
    
    xaxis = "percentage"
    lines = "q"
    plotCombinations(xaxis, lines, const, "avg_en", savedir, "avg_en", ylim=(0,None))
    try:
        _q = max(const["q"])
    except TypeError:
        _q = const["q"] 
    maxSpeed = analytics.equilSpeedPropulsing(const, q=_q)
    plotCombinations(xaxis, lines, const, "avg_vel", savedir, "avg_vel", ylim=(0,maxSpeed))
    minLength = analytics.minPathLength(const)
    #plotCombinations(xaxis, lines, const, "distance_from_goal", savedir, "distance_from_goal", ylim=(0,minLength))
    #plotCombinations(xaxis, lines, const, "avg_path_length", savedir, "avg_path_length", ylim=(0,2100))
    #plotCombinations(xaxis, lines, const, "avg_path_length_err", savedir, "avg_path_length_err", ylim=(0,None))
    plotCombinations(xaxis, lines, const, "avg_ci", savedir, "avg_ci", ylim=(0,None))
    plotGlobal(xaxis, lines, const, "success_ratio", savedir, "success_ratio"+constants.graphics_ending, ylim=(0,1))
    plotGlobal(xaxis, lines, const, "success_ratio_a", savedir, "success_ratio_a"+constants.graphics_ending, ylim=(0,1))
    plotGlobal(xaxis, lines, const, "success_ratio_m", savedir, "success_ratio_m"+constants.graphics_ending, ylim=(0,1))
    
    
    #plotFitness(xaxis, lines, const, savedir, "fitness"+constants.graphics_ending)
    
    if simfile=="with-without-i.py":
        plotGlobal2factors(xaxis, ["q", "enable_interaction"], const, "success_ratio", savedir, "success_ratio"+constants.graphics_ending, ylim=(0,1))
        plotGlobal2factors(xaxis, ["q", "enable_interaction"], const, "success_ratio_a", savedir, "success_ratio_a"+constants.graphics_ending, ylim=(0,1))
    
    
    if simfile=="densities-all.py":
        plotGlobal(xaxis, "maze", const, "success_ratio", savedir, "success_ratio2"+constants.graphics_ending, ylim=(0,1), disableVarInLegend=True)
        plotGlobal(xaxis, "maze", const, "success_ratio_a", savedir, "success_ratio2_a"+constants.graphics_ending, ylim=(0,1), disableVarInLegend=True)
        plotGlobal(xaxis, "maze", const, "avg_ci", savedir, "avg_ci2"+constants.graphics_ending, ylim=(0,1), disableVarInLegend=True)
        plotGlobal(xaxis, "maze", const, "avg_ci_a", savedir, "avg_ci2_a"+constants.graphics_ending, ylim=(0,1), disableVarInLegend=True)
        plotGlobal(xaxis, "maze", const, "avg_ci_a_s", savedir, "avg_ci2_a_s"+constants.graphics_ending, ylim=(0,1), disableVarInLegend=True)
        plotGlobal(xaxis, "maze", const, "avg_ci_a_us", savedir, "avg_ci2_a_us"+constants.graphics_ending, ylim=(0,1), disableVarInLegend=True)
    
    print "Done."

