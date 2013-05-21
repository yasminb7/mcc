import numpy as np
import math
import logging, os.path, shutil, re, itertools, importlib, time, string
from constants import confpackage, resourcespath, videoshellscript, ignore
import subprocess

def debug(*args, **kwargs):
    """Log a string or several at DEBUG level."""
    s = ""
    for a in args:
        s += str(a) + " "
    logging.debug(s)

def info(*args, **kwargs):
    """Log a string or several at INFO level."""
    s = ""
    for a in args:
        s += str(a) + " "
    logging.info(s)

def error(*args, **kwargs):
    """Log a string or several at ERROR level."""
    s = ""
    for a in args:
        s += str(a) + " "
    logging.error(s)

def norm(v):
    """Calculate the Euclidean norm of a two-dimensional vector."""
    return math.sqrt(np.sum(v*v))

def normsq(v):
    """Calculate the squared norm."""
    v = np.asarray(v)
    return np.sum(v*v)
    
def setup_logging_base():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return ch

def setup_logging_sim(logfilepath):
    logger = logging.getLogger()
    
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    
    fh = logging.FileHandler(logfilepath, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return fh

def remove_logging_sim(fh):
    logger = logging.getLogger()
    logger.removeHandler(fh)

def readConst(directory, simfile):
    """Return the parameter settings containted in the variable const in a file in the folder sim/."""
    #TODO: improve how the name is found (had better be outside of this function)
    name = simfile[:-3]
    m = confpackage + name
    mod = importlib.import_module(m, package=None)
    const = mod.const
    return const

def renamedir(old, new):
    if os.path.exists(new):
        shutil.rmtree(new)
    os.rename(old, new)

def validSimfile(filename):
    """Check that a given file is a valid simfile. Ignore if it starts with _ or is in the ignore list."""
    atoms = filename.rsplit('.', 1)
    if filename[0] == '_' or atoms[-1] != 'py' or filename in ignore:
        return False
    return True

def getSimfiles(simdir):
    """Get all valid simfiles in simdir."""
    #list files
    files = os.listdir(simdir)
    filestocheck = []
    for f in files:
        if validSimfile(f):
            filestocheck.append(f)
    return filestocheck

def getDirTimestamp(directory):
    """Get the timestamp of a given simulation."""
    stampfile = None
    files = os.listdir(directory)
    for filename in files:
        if re.match("[0-9]+\-[0-9]+\-[0-9]+", filename):
            stampfile = filename
    if stampfile is None:
        return None
    path = os.path.join(directory, stampfile)
    timestamp = os.stat(path).st_mtime
    return timestamp

def verifySimulationNecessary(constfilename, resultsfolder):
    """Run the simulation if there are no results corresponding to the sim file or
    if the sim file is newer than the results.
    """
    if not os.path.exists(resultsfolder):
        return True
    else:
        simtimestamp = os.stat(constfilename).st_mtime
        dirstamp = getDirTimestamp(resultsfolder)
        if dirstamp is None:
            return True
        elif simtimestamp > dirstamp:
            return True
    return False

def savetimestamp(folder):
    """Save the timestamp for a given results folder."""
    timestampfilename = time.strftime("%H-%M-%S",time.localtime())
    timestamp=file(getResultsFilepath(folder, timestampfilename) ,'w')
    timestamp.close()

def deleteTimestamp(folder):
    timestamp = getDirTimestamp(folder)
    if timestamp is not None:
        files = os.listdir(folder)
        for filename in files:
            if re.match("[0-9]+\-[0-9]+\-[0-9]+", filename):
                os.remove(os.path.join(folder, filename))

def updateTimestamp(folder):
    if os.path.exists(folder):
        deleteTimestamp(folder)
        savetimestamp(folder)
        return True
    else:
        return False

def getResultsDir(filename):
    """From a given simfile, create a name for a folder in results that is obtained by removing the file ending."""
    if ".py" in filename:
        return filename[0:-3] + "/"
    else:
        return filename + "/"

#natural sorting of a list
#meaning for example that "a10" comes after "a9"
#the standard sorting in python gives "a9", "a10" because 1<9
def natsort(list_, separator="_", lastisint=False):
    """Function not used as of now."""
    splitlist = sorted([map(floatifpossible, s.split(separator)) for s in list_])
    if lastisint:
        for item in splitlist:
            item[-1] = int(item[-1])
    newlist = [separator.join(map(str, el)) for el in splitlist]
    return newlist

def floatifpossible(s):
    r = ""
    try:
        r=float(s)
    except ValueError:
        r=s
    return r

def getFolders(searchpath, pattern):
    """Get a list of all the folders matching pattern."""
    root = os.getcwd()
    os.chdir(root)
    dirs = os.listdir(searchpath)
    sweepdirs = list()
    for directory in dirs:
        p = re.compile(pattern)
        if p.search(directory):
            sweepdirs.append(searchpath + directory)
    return sweepdirs

def scaleToMax(newmax, arr):
    """Make sure the maximum value of all the pixel in an array is newmax."""
    currentmax = np.amax(arr)
    return (newmax/currentmax) * arr

def reduceFolderToRegexp(folder):
    r = folder
    if folder[-2] == '_':
        r = folder[:-2]
    return r

def makeFolderEmpty(folder):
    """Remove all the files in a folder."""
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception, e:
            print e

def unravel(const):
    """Unravels a given const file along a certain axis, creating as many const variables
    as can be obtained by the parameter settings in the file.
    """
    constlist = []
    factors = [const[key] for key in const["factors"]]
    prod = itertools.product(*factors)
    #add corresponding const dictionaries
    for p in prod:
        newconst = const["get"](p)
        constlist.append(newconst)
    return constlist

def unravel_fixedaxis(const, axis, exclude=None):
    """
    Unravels a given const file along a certain axis.
    Returns a list of lists:
    - the outer list containing a tuple consisting of the inner list and an appropriate name (for files, folders)
    - the inner list containing the const dictionaries arranged as given by "axis".
    """
    assert axis in const["factors"], "You picked an axis that doesn't exist."
    if len(const["factors"])==1:
        name = const["name"] % "#"
        return [(unravel(const), name)]
    constlist = []
    factors = [const[key] if (key!=axis and key!=exclude) else [None] for key in const["factors"]]
    a_values = const[axis]
    prod = itertools.product(*factors)
    #add corresponding const dictionaries
    for p_ in prod:
        axislist = []
        for a_ in a_values:
            p = [factor if factor is not None else a_ for factor in p_]
            p = tuple(p)
            newconst = const["get"](p, exclude="repetitions")
            axislist.append(newconst)
        specP = [factor if factor is not None else "#" for factor in p_]
        if exclude=="repetitions" and exclude in const["factors"]:
            name = string.replace(const["name"], "_r%s", "")
            specP = tuple([specP[i] for i in range(len(specP)-1)])
        else:
            name = const["name"]
        name = name % tuple(specP)
        constlist.append((axislist, name))
    return constlist

def applyFilter(constlist, name, valuelist):
    """Given a list of const variables, return another list containing only the ones fulfilling
    a given criteria, namely the parameter name has to have one of the values in valuelist.
    """
    r = [c for c in constlist if c[name] in valuelist]
    return r

def retainCompleteDataset(const):
    return ( const["simulations_with_complete_dataset"] <= const["repetitions"] + 1 )

def getResultsFilepath(*pathelements):
    """Given one or several folders and possibly a filename return the complete path."""
    path = os.path.join(*pathelements)
    return path

def ensure_dir(f):
    """Make sure that the directory f exists by creating it if necessary."""
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

def getGradient(a):
    """Return the gradient as given by the NumPy gradient function."""
    old_err_state = np.seterr(divide='ignore', invalid='ignore')
    gx, gy = np.gradient(a)
    np.seterr(**old_err_state)
    return np.array((gx, gy))

def getOriginalGradient(mazefilename, maze=None):
    """At the beginning of the simulation, read the gradient for a maze from a file
    (or, if the file doesn't exist, create it and save the gradient in it).
    """
    assert os.path.exists(mazefilename)
    mazegradfilename = "%s.grad" % mazefilename
    if os.path.exists(mazegradfilename):
        with open(mazegradfilename, "rb") as grfile:
            gradient = np.load(grfile)
    elif maze is not None:
        gradient = getGradient(maze)
        with open(mazegradfilename, "wb") as grfile:
            np.save(grfile, gradient)
    return gradient

def copyAnimScript(destination):
    """For ease of use copy shell script that creates video right into the results folder."""
    source = os.path.join(resourcespath, videoshellscript)
    destination = getResultsFilepath(destination, videoshellscript)
    shutil.copy(source, destination)
    
def exec_silent_command(command):
    """Execute a given command, suppressing all output."""
    with open(os.devnull, "wb") as fh:
        subprocess.Popen(command, stdout=fh, stderr=fh)

def removeFilesByEnding(path, ending):
    """Delete all files in a folder that have a given ending."""
    oldpath = os.getcwd()
    files = os.listdir(path)
    n = len(ending)
    os.chdir(path)
    for f in files:
        if len(f)>n and f[-n:]==ending:
            deleteFile(os.path.join(path, f))
    os.chdir(oldpath)

def deleteFile(path):
    if os.path.exists(path):
        os.remove(path)

def concentration(i, j, gradientcenter):
    c = gradientcenter
    field = (i-c[0]) * (i-c[0]) + (j-c[1]) * (j-c[1])
    return -1.0 * field
