.. _run:

How to run simulations
======================

Once you've set up your simulation as described in :doc:`setupsim`, you can run it by launching the main script.

>>> python main.py

The program then reads all the simulation descriptions from the *sim/* directory, ignoring those whose filename starts with _.
For each of the descriptions it generates a list of all necessary simulations and checks if they've already been run.

If for one of these simulations generated, there is already a result directory containing a valid timestamp (i.e., a file with a name like 08-22-52), and the timestamp file is newer than the modification date of the simulation description in *sim/*, then the program assumes that the simulation has been run for the relevant parameters and no rerun is needed. It therefore skips that particular simulation.

On the other hand this means that if you modify your simulation description in *sim/* then **all** the simulations described by it will be rerun, whether you intended that or not. The solution is to renew the timestamps by calling

>>> python renewTimestamps.py

with the correct ``currentsim`` setting in :py:mod:`constants`.
Once you've done this, you can launch ``main.py`` and it will run only the simulations for which the results don't exist yet.
