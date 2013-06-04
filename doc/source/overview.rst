Overview
========

So what does ``mcc`` do for you?

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


