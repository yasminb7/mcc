'''
Created on 15.10.2012

@author: frederic
'''

import scipy as sp
import string
from os.path import join
import matplotlib
import matplotlib.pyplot as plt
import itertools
from utils import info

mydpi = 300

colors = ['r', 'g', 'b', 'y']
#matplotlib.rcParams['font.size']=12.0
#matplotlib.rcParams['font.family']="serif"
#matplotlib.rcParams['font.family']="serif"
matplotlib.rcParams['font.serif']=["Latin Modern", "Computer Modern Roman", "Times"]
#matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['savefig.dpi']=100.

mymarkersize = 7
#formats = ["bo", "gs", "r^", "yo", "ks", ""]
#formats = itertools.cycle(formats)
mymarkers = 'os>^+*'

def plot(x, y, labels = None, xlabel = None, ylabel = None, xlabels=None, title=None, legend=None, showGrid=False, folder=None, savefile=None):
    assert len(x)==len(y), "x and y don't have the same length"
    assert len(x)>0 and type(x)==list
    markers = itertools.cycle(mymarkers)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(showGrid)
    handles = []
    for i, m in enumerate(x):
        handles.append(ax.plot(x[i], y[i], linestyle=':', marker='s')[0])
        #ax.set_xlim([-0.8,0.4])
        #ax.set_ylim([-0.1,0.1])
        #That label stuff doesn't yet work as it should.
        if labels is not None:
            ax.text(x[i][0], y[i][0], labels[0], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    if legend is not None:
        ax.legend(handles, legend, loc=0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if folder is None:
        folder = ""
    if savefile is not None:
        plt.savefig(join(folder, savefile))
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()

def plot_histogram(data, xlabel = None, ylabel = None, title = None, folder=None, savefile=None, nbins=20, **plotargs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    # the histogram of the data
    n, bins, patches = ax.hist(data, nbins, facecolor='green', alpha=0.75)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(True)
    if title is not None:
        ax.set_title(title)
    if "xlim" in plotargs.keys():
        ax.set_xlim(plotargs["xlim"])
    if "ylim" in plotargs.keys():
        ax.set_ylim(plotargs["ylim"])
    if folder is None:
        folder = ""
    if savefile is not None:
        plt.savefig(join(folder, savefile))
        info(savefile + " written.")
    plt.close()

def errorbars(x, y, y_bars=None, labels = None, xlabel = None, ylabel = None, xlabels=None, title=None, legend=None, showGrid=False, folder=None, savefile=None, **plotargs):
    assert len(x)==len(y), "x and y don't have the same length"
    markers = itertools.cycle(mymarkers)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(showGrid)
    handles = []
    assert len(x)==len(y)
    if y_bars is not None:
        assert len(y)==len(y_bars)
    #assert len(x)<len(formats)
    for i, m in enumerate(x):
        marker = markers.next()
        #handles.append(ax.errorbar(x[i], y[i], fmt = formats[i], markersize = mymarkersize, yerr=y_bars[i])[0])
        if y_bars is None:
            handles.append(ax.errorbar(x[i], y[i], markersize = mymarkersize, fmt=':%s' % marker)[0])
        else:
            handles.append(ax.errorbar(x[i], y[i], yerr=y_bars[i], markersize = mymarkersize, fmt=':%s' % marker)[0])
        #ax.set_xlim([-0.8,0.4])
        #ax.set_ylim([-0.1,0.1])
        #That label stuff doesn't yet work as it should.
        if labels is not None:
            ax.text(x[i][0], y[i][0], labels[0], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    if legend is not None:
        ax.legend(handles, legend, loc=0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if "xlim" in plotargs.keys():
        ax.set_xlim(plotargs["xlim"])
    if "ylim" in plotargs.keys():
        #_ylim = plotargs["ylim"]
        #if len(_ylim)==2:
        ax.set_ylim(plotargs["ylim"])
        #else:
            #ax.set_ylim(ymin)
    if folder is None:
        folder = ""
#    if setBW is True:
#    setFigLinesBW(fig)
    if savefile is not None:
        savefilepath = join(folder, savefile)
        plt.savefig(savefilepath)
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()

def bars_stacked(y0, color0, y1, color1, left, width, y_bars=None, labels = None, xlabel = None, ylabel = None, xlabels=None, title=None, legend=None, showGrid=False, folder=None, savefile=None, **plotargs):
    assert len(y0)==len(y1), "y0 and y1 don't have the same length"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(showGrid)
    handles = []
    
    h0 = ax.bar(left, y0, width, color=color0, label="successful")
    h1 = ax.bar(left, y1, width, bottom=y0, color=color1, label="unsuccessful")
    handles.append(h0)
    handles.append(h1)
    ax.legend()
    if legend is not None:
        ax.legend(handles, legend, loc=0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if "xlim" in plotargs.keys():
        ax.set_xlim(plotargs["xlim"])
    if "ylim" in plotargs.keys():
        ax.set_ylim(plotargs["ylim"])
    if folder is None:
        folder = ""
#    if setBW is True:
#        setFigLinesBW(fig)
    if savefile is not None:
        savefilepath = join(folder, savefile)
        plt.savefig(savefilepath)
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()
    
def scatter(xs, ys, amoeboids, mesenchymals, successful, labels = None, xlabel = None, ylabel = None, xlabels=None, title=None, legend=None, showGrid=False, folder=None, savefile=None, **plotargs):
    assert len(xs)==len(ys), "y0 and y1 don't have the same length"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(showGrid)
    handles = []
    
    s_m = sp.logical_and(successful, mesenchymals)
    u_m = sp.logical_and(~successful, mesenchymals)
    s_a = sp.logical_and(successful, amoeboids)
    u_a = sp.logical_and(~successful, amoeboids)
    
    h0 = ax.scatter(xs[s_m], ys[s_m], color='g', marker='o', s=7, label="successful mesenchymals")
    h1 = ax.scatter(xs[u_m], ys[u_m], color='g', marker='^', s=7, label="unsuccessful mesenchymals")
    h2 = ax.scatter(xs[s_a], ys[s_a], color='b', marker='o', s=7, label="successful amoeboids")
    h3 = ax.scatter(xs[u_a], ys[u_a], color='b', marker='^', s=7, label="unsuccessful amoeboids")
    handles.append(h0)
    handles.append(h1)
    handles.append(h2)
    handles.append(h3)    
    ax.legend()
    if legend is not None:
        ax.legend(handles, legend, loc=0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if "xlim" in plotargs.keys():
        ax.set_xlim(plotargs["xlim"])
    if "ylim" in plotargs.keys():
        ax.set_ylim(plotargs["ylim"])
    if folder is None:
        folder = ""
#    if setBW is True:
#        setFigLinesBW(fig)
    if savefile is not None:
        savefilepath = join(folder, savefile)
        plt.savefig(savefilepath)
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()
    
def scattercisr(xs, ys, amoeboids, mesenchymals, labels = None, xlabel = None, ylabel = None, xlabels=None, title=None, legend=None, showGrid=False, folder=None, savefile=None, **plotargs):
    assert len(xs)==len(ys), "y0 and y1 don't have the same length"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(showGrid)
    handles = []
    
    h0 = ax.scatter(xs[mesenchymals], ys[mesenchymals], color='g', marker='o', s=7, label="mesenchymals")
    h1 = ax.scatter(xs[amoeboids], ys[amoeboids], color='b', marker='o', s=7, label="amoeboids")
    handles.append(h0)
    handles.append(h1)
    ax.legend(loc=10)
    if legend is not None:
        ax.legend(handles, legend, loc=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if "xlim" in plotargs.keys():
        ax.set_xlim(plotargs["xlim"])
    if "ylim" in plotargs.keys():
        ax.set_ylim(plotargs["ylim"])
    if folder is None:
        folder = ""
#    if setBW is True:
#        setFigLinesBW(fig)
    if savefile is not None:
        savefilepath = join(folder, savefile)
        plt.savefig(savefilepath)
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()


def plotlines(x, y, labels = None, xlabel = None, ylabel = None, xlabels=None, ylimits=None, title=None, legend=None, showGrid=False, folder=None, savefile=None):
    assert len(x)==len(y), "x and y don't have the same length"
    assert len(x)>0 and type(x)==list
    markers = itertools.cycle(mymarkers)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(showGrid)
    handles = []
    for i, m in enumerate(x):
        handles.append(ax.plot(x[i], y[i], linestyle='-', marker=None)[0])
        #ax.set_xlim([-0.8,0.4])
        if ylimits is not None:
            ax.set_ylim(ylimits)
        #That label stuff doesn't yet work as it should.
        if labels is not None:
            ax.text(x[i][0], y[i][0], labels[0], horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    if legend is not None:
        ax.legend(handles, legend, loc=0)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if folder is None:
        folder = ""
    if savefile is not None:
        plt.savefig(join(folder, savefile))
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()
    
def plotlines_vert_subplot(x, y, x2, y2, xlabel = None, ylabel = None, xlabel2 = None, ylabel2 = None, ylimits=None, ylimits2=None, title=None, legend=None, legend2=None, showGrid=False, folder=None, savefile=None):
    assert len(x)==len(y), "x and y don't have the same length"
    assert len(x)>0 and type(x)==list
    colors = matplotlib.rcParams['axes.color_cycle']
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.grid(showGrid)
    handles = []
    for i in range(len(x)):
        handles.append(ax.plot(x[i], y[i], linestyle='-', marker=None, c=colors[i])[0])
        if ylimits is not None:
            ax.set_ylim(ylimits)
    if legend is not None:
        #ax.legend(handles, legend, loc=0)
        ax.legend(handles, legend)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    
    handles2 = []
    ax2 = fig.add_subplot(212)
    for i in range(len(x)):
        handles2.append(ax2.plot(x[i], y2[i], linestyle='-', marker=None, c=colors[i])[0])
        if ylimits2 is not None:
            ax2.set_ylim(ylimits2)
    if legend2 is not None:
        #ax2.legend(handles2, legend_twin, loc=0)
        ax2.legend(handles2, legend2)
    if xlabel2 is not None:
        ax2.set_xlabel(xlabel2)
    if ylabel2 is not None:
        ax2.set_ylabel(ylabel2)
    
    if title is not None:
        ax.set_title(title)
    if folder is None:
        folder = ""
    if savefile is not None:
        plt.savefig(join(folder, savefile), dpi=mydpi)
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()

def plotlines_twinaxes(x, y, y_twin, labels = None, xlabel = None, ylabel = None, xlabels=None, ylimits=None, ylimits_twin=None, title=None, legend=None, legend_twin=None, showGrid=False, folder=None, savefile=None):
    assert len(x)==len(y), "x and y don't have the same length"
    assert len(x)>0 and type(x)==list
    colors = itertools.cycle(matplotlib.rcParams['axes.color_cycle'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.grid(showGrid)
    handles = []
    for i in range(len(x)):
        color = colors.next()
        handles.append(ax.plot(x[i], y[i], linestyle='-', marker=None, c=color)[0])
        if ylimits is not None:
            ax.set_ylim(ylimits)
    handles2 = []
    ax2 = ax.twinx()
    for i in range(len(x)):
        color = colors.next()
        handles2.append(ax2.plot(x[i], y_twin[i], linestyle='-', marker=None, c=color)[0])
        if ylimits_twin is not None:
            ax2.set_ylim(ylimits_twin)
    if legend is not None:
        #ax.legend(handles, legend, loc=0)
        ax.legend(handles, legend)
    if legend_twin is not None:
        #ax2.legend(handles2, legend_twin, loc=0)
        ax2.legend(handles2, legend_twin)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if folder is None:
        folder = ""
    if savefile is not None:
        plt.savefig(join(folder, savefile))
        info(savefile + " written.")
    if savefile is None:
        plt.show()
    plt.close()

def texEscape(s):
    specials = ["_", "#", "$", "%", "&", "{", "}"]
    for spec in specials:
        s = string.replace(s, spec, '\\' + spec)
    return s

def createLegend(s, varnames):
    replacements = {"mass": "m", "gamma" : "$\gamma$"}
    legend = ""
    elements = string.split(s, "_")
    for el in elements:
        for var in varnames:
            idx = string.find(el, var) 
            if idx>=0:
                text = var
                if var in replacements.keys():
                    text = replacements[var]
                legend += text + " " + el[idx+len(var):] + " "
    if legend=="":
        return None
    else:
        return legend

def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be 
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

    COLORMAP = {
        'b': {'marker': None, 'dash': (None,None)},
        'g': {'marker': None, 'dash': [5,5]},
        'r': {'marker': None, 'dash': [5,3,1,3]},
        'c': {'marker': None, 'dash': [1,3]},
        'm': {'marker': None, 'dash': [5,2,5,2,5,10]},
        'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }

    for line in ax.get_lines():
        origColor = line.get_color()
        line.set_color('black')
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)

if __name__ == "__main__":
    import scipy as sp
    x = sp.arange(0, 2*sp.pi, 0.1)
    y = sp.cos(x)
    z = sp.sin(x)
    plot([x, x], [y, z])
    