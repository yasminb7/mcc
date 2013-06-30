Analyse the results
===================

.. _dataset-arrays:

What exactly do the results contain?
------------------------------------

When you run the simulations, the resulting data will be saved in a subfolder of `results/`, the name of the subfolder being the one you defined in your *sim*-file. In the following I'll describe what files that folder contains (names may vary).

*Note:* `There may well be additional files, but these depend on the output of your` ``statistics.py`` `so I won't describe them here.`

**18-39-19**
    Timestamp file: Contains the date/time the results were produced.

**A_delays.npy**
    `You won't need this.`
    Shape: N

    The (random) values resulting from the distribution characterized by the `orientationdelay` parameter in the *sim*-file.

**A_direction_angles.npy**
    `You won't need this.`
    Shape: N

    The values of the angles characterizing the direction of movement.

**A_eating.npy**
    Shape: T*N

    Indicating if at a certain timestep a certain agent is degrading the ECM or not.

**A_energies.npy**
    Shape: T * N

    The energy of each agent at each timestep.

**A_periods.npy**
    `You won't need this.`
    Shape: N

    The (random) values resulting from the distribution characterized by the `orientationperiod` parameter in the *sim*-file.


**A_positions.npy**
    Shape: T * N * 2

    The position (two dimensions) of each agent at each timestep.

**A_positions.npy.shape.pickle**
    A ``pickle`` file (see Python documentation) containing the shape of the positions array (for convenience).

**A_statechanges.npy**
    `You won't need this`
    Shape: N

    Used only during simulations. Contains the times at which agents switched from a state (like `MOVING`) to another (like `ORIENTING`).

**A_states.npy**
    Shape: T * N

    Contains the states for any given agent at any given timestep.

**A_times.npy**
    Shape: T

    For convenience. Contains a list of all the simulation times at each timestep. Like 0.0, 0.1, 0.2, ...

**A_types.npy**
    Shape: N

    Contains an integer indicating the type of each agent. For historical reasons, you need to be careful here. If it contains 0 and 1, then 0 stands for amoeboid and 1 for mesenchymal. If it contains 1 and 2, then 1 stands mesenchymal and 2 stands for amoeboid. This is stupid, I know.

**A_velocities.npy**
    Shape: T * N * 2

    Contains the velocities of each agent at each timestep. Strictly speaking it's probably not needed because the velocity is the derivative of the position which we already have.

**A_velocities.npy.shape.pickle**
    For convenience: Contains the shape of the velocities array.

**finalmaze.npy**
    `You probably won't need this.`

    Contains the maze describing the array at the last timestep.

**finalstats.pickle**
    Contains a pickled dictionary describing some statistics at the end of the simulation.

**finalstats.txt**
    Contains the final statistics written out in text.

**frames/**
    A directory created by ``graphics.py`` where the single frames used to create the video are stored.

**logfile_create_video.txt**
    A log file documenting the process of creating a video. Not likely to be interesting.

**logfile.txt**
    A log file containing messages from the simulation run.

**paths.png**
    Created by ``graphics.py`` and contains the plot of the trajectories.

Create statistics and plots
---------------------------

Once all the simulations are run, you can launch the analysis by running

>>> python statistics.py

You may need to adapt :py:mod:`statistics` depending on what you have in mind with your simulation and the information you want to get out of it.

Alternative: Get results in a CSV format
----------------------------------------

If you prefer to do further analysis with other types of software (MATLAB, Excel etc.), you can use the provided script :doc:`code/ds2csv`.

Alternative: Load simulation data for further use in a Python program
---------------------------------------------------------------------

The script :doc:`code/exploreDataset` provides a simple skeleton to

* explore your own ideas with NumPy on realistic data
* try out NumPy functions work
* find extrema or average values etc. in suspicious datasets

