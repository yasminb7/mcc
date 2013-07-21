Overview
========

 **Note:** Most of the files in the folder `sim/` are from the work during my thesis. After it, I worked mainly on `maze-easy-ar.py`, so this file is probably where you'll want to start.

What does it do for you?
------------------------

It lets you describe a bunch of simulations you want to run by creating a file with a certain structure in the `sim/` folder.
Assume you have a parameter `q` and a parameter `w` that you want to vary in order to see their combined effect.
You can specify this as ``q = [5, 6, 7]`` and ``w = [0.3, 0.6]`` and ``mcc`` interprets this as *all possible combinations* of `q` and `w` (i.e., the cartesian product). For these particular values of `q` and `w` you'll get the combinations (5, 0.3), (5, 0.6), (6, 0.3), (6, 0.6) etc.

When you launch it, ``mcc`` then checks the `sim/` folder to determine all the simulations it is expected to run.
If some of the simulation results already exist (for example because of an interrupted simulation run the day before), it recognizes that fact and launches just the ones that are needed to complete the simulations described in the folder `sim/`.

The results of each simulation is stored in the folder `results/` with a name (that you determine) that should make it easy to recognize what the intent of that particular simulation was.
``mcc`` can either:

* save *all* the data resulting from a simulation (i.e., each agent's position, velocity etc. at each timestep); or
* save only some statistics (simply called `finalstats`), that it calculated at the end of the simulation using the data produced.

Usually you will want to run a particular simulation repeatedly so you can average over a certain value in `finalstats`.
If you save all simulation data for each repetition, you will be filling your harddisk with useless data.
On the other hand, if you don't save the entire simulation dataset for any of the simulations, it will be impossible to reconstruct i.e. plots of trajectories.
To solve this dilemma, you can specify that a certain number of repetitions (usually 1) will save the complete dataset, while the rest will merely save the `finalstats`.

What's all that data that came with the `mcc` code?
---------------------------------------------------

The data that's included is there either

* because we've used it to produce my thesis (which should also be among the files you got)
* or because it's data we were currently working on when I handed over the project.

In the case of data used in my thesis, :ref:`figure-to-sim` points you to the *sim*-files that were used to produce each relevant figure in my thesis.


.. _figure-to-sim:

How do the figures in the thesis correspond to the data you received?
---------------------------------------------------------------------

* Figure 4 represents the *success rate* plot produced from ``maze-easy.py``, which, when run with

>>> python main.py
>>> python statistics.py

produces the plots in ``results/maze-easy/``. The data is saved in the folders with names like ``results/maze-easy-pM0.2_q1.0_r2/``.

* Figures 5 and 6 are the same as figure 4 for ``maze-medium.py`` and ``maze-hard.py``, respectively.

* Figure 7 represents the trajectory plot from the simulation whose data is stored in ``results/maze-align-easy_pM0.3_q0.6_r0``.

* Figures 8, 9 and 10 are the same as figures 4, 5 and 6 except for added alignment interaction. The simulation parameters are given in ``maze-align-easy.py``, ``maze-align-medium.py`` and ``maze-align-hard.py``. The produced data follows the same pattern: ``results/maze-align-easy-pM*.*_q*.*_r*/`` and so on.

* Figures 11, 12, 13 are from the sim-files ``densities-1.py``, ``densities-2.py`` and ``densities-3.py``. It's very unlikely that you'll ever need these results again, so I didn't include them in the data. If you do need them you can just rerun the simulations, it shouldn't take more than a few hours.

* Figures 14 and 15 were generated using ``densities-all.py``. The data is in the folders following the same pattern for names.


Notice that the file ``statistics.py`` has evolved over time and may not produce exactly the same plots anymore. Plots that have become less important have had the relevant code commented out.

