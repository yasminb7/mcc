"""
This script is intended to help you analyse datasets in a Python interpreter.
It takes a stored :py:class:`.Dataset` in a given folders and loads it.

Use this if you want to, for example, do a quick search for extrema in an array, or if you want to try out NumPy functions.

It can be helpful to use a debugger to check the content of variables at certain points in the program.
I recommend Eclipse with the PyDev plugin.
"""

import os

from mcclib.classDataset import load
from mcclib.classDataset import Dataset
import mcclib.constants as constants

folder = "maze-easy-ar_pM0.1_q1.0_r0"
resultsbase = constants.resultspath
path = os.path.join(resultsbase, folder)
ds = load(Dataset.ARRAYS, path, "A_", 0.1, readOnly=True)

#Write your own commands here