renewTimestamps.py
==================

As explained in :ref:`run`, ``main.py`` determines which simulations to run by comparing the modification date of a sim-file with the timestamps of all the folders in ``results/`` that the sim-file generates.

Therefore, if you modify the sim-file, the program will assume that all simulations run earlier have become invalid and will overwrite them.
However, in some situations (like when you add a few repetitions), the earlier simulations haven't actually become invalid.

In that case the trick is to

* modify the sim-file,
* make sure that sim-file is the one specified in the variable ``mcclib/constants.py``
* and then run

>>> python renewTimestamps.py

When you run ``renewTimestamps.py``, it simply goes through all the folders that the sim-file can generate and, if they exist, updates the timestamp. As a consequence, when you run

>>> python main.py

again, these simulations will appear to have been created by the current sim-file, and will not be run again uselessly.

Note that the responsibility for knowing when to run ``renewTimestamps.py`` is yours. If you run it at the wrong moment it can mess up your simulation results and invalidate any conclusions made from them, so be careful.
