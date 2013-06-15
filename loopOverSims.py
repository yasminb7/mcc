from mcclib import utils, constants
import os, csv
from mcclib.classDataset import Dataset, load

sims = [constants.currentsim]

for currentsim in sims:    
    const = utils.readConst(constants.simdir, currentsim)
    constlist = utils.unravel(const)
    constlist = utils.applyFilter(constlist, "q", [1.0])
    cwd = os.getcwd()
    print "%s folders" % len(constlist)
    outfilename = const["name"] + ".csv"
    with open(outfilename, "wb") as _csvOut:
        csvOut = csv.writer(_csvOut, dialect=csv.excel)
        csvOut.writerow(["% mesenchymals", "repetition", "agent", "type", "successful", "time to target", "CI"])
        myData = dict()
        for i, c in enumerate(constlist):
            if c["percentage"] not in myData:
                myData[c["percentage"]] = dict()
            if c["q"] not in myData[c["percentage"]]:
                myData[c["percentage"]] = dict()
            
            myData[c["percentage"]]["lines"] = list()
            
            path = os.path.join(constants.resultspath, c["name"])
            rep = c["repetitions"]
            ds = load(Dataset.AMOEBOID, path, fileprefix="A_", dt=0.1, readOnly=True)
            assert ds is not None
            TYPE_AMOEBOID = 0 if rep<=9 else 2
            TYPE_MESENCHYMAL = 1 if rep<=9 else 1
            types = ["A" if t==TYPE_AMOEBOID else "M" for t in ds.types]
            N = len(types)
            index = range(N)
            pM = c["percentage"]
            
            path = os.path.join(cwd, constants.resultspath, c["name"])
            print "%03d/%s: Reading %s:" % (i+1, len(constlist), c["name"])
            csvfilename = utils.getResultsFilepath(constants.resultspath, c["name"], constants.individual_CI_filename)
            with open(csvfilename, "rb") as _csvFile:
                csvFile = csv.reader(_csvFile, dialect=csv.excel)
                ci = csvFile.next()
            
            tttfilename = utils.getResultsFilepath(constants.resultspath, c["name"], constants.individual_ttt_filename)
            with open(tttfilename, "rb") as _csvFile:
                csvFile = csv.reader(_csvFile, dialect=csv.excel)
                ttt = csvFile.next()
                ttt = map(lambda x: x if x!="2500.0" else "", ttt)
    
            successfilename = utils.getResultsFilepath(constants.resultspath, c["name"], constants.individual_success_filename)
            with open(successfilename, "rb") as _csvFile:
                csvFile = csv.reader(_csvFile, dialect=csv.excel)
                success = csvFile.next()
                success = map(lambda x: 1 if x=="True" else 0, success)
            
            outData = zip(N*[pM], N*[rep], index, types, success, ttt, ci)
            for line in outData:
                csvOut.writerow(line)
            
print "Done."
