#!/usr/bin/env python

import platform
import numpy
import scipy
import matplotlib

print "Architecture: %s " % str(platform.architecture())
print "System: %s " % platform.system()
print "Release: %s " % platform.release()
print "Version: %s " % platform.version()
print
print "Python: %s " % platform.python_version()
print "NumPy: %s " % numpy.__version__
print "SciPy: %s " % scipy.__version__
print "Matplotlib: %s" % matplotlib.__version__
