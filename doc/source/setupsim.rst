Set up the simulations
======================

If you want just one simulation with all the parameter values fixed, read :ref:`simple-sim`.
If you want simulations over a range of values (or repeated a few times), read :ref:`sim-with-range`.
If you want to find out more about the possible parameters first, read :ref:`sim-parameters`.

.. _simple-sim:

A simple simulation
-------------------

To run one single simulation, make a copy the file ``prototype-single.py`` in the ``sim/`` folder and give it a meaningful name.

 Note: If the name starts with an underscore (``_``), the file will be ignored.

Then modify the parameters according to your liking.

 Note: One parameter you absolutely *should* modify is `name`, because it defines the name of the subfolder of ``results/`` where the results of your simulation will reside.

The various parameters are explained in the comments of the file, as well as in the documentation under :ref:`sim-parameters`.

When done, continue with :ref:`run`.

.. _sim-with-range:

Simulations including a range of values
---------------------------------------

For simulations that include varying parameters (e.g., several values of `q`), the process is almost the same, and the same remarks as in the above section apply, so please read that first if you haven't read it yet.

Copy the file ``prototype.py`` and give it a name.
Again, modify the file to your liking.

The interesting part here are the variables that take several values. *The names of these variables have to be put in the list* `factors` *with their names exactly as they appear in the configuration.*

 Assume you want to have several values of `q`, and for each you want a few repetitions of that simulation. In that case the line containing `factors` has to look like: `"factors" : ["q", "repetitions"],`.

Doing so will cause the program to generate all possible values (i.e., the Cartesian product) of these values and then run all these simulations.

After you define the variable `factors`, you have to define the actual *values* of these so-called factors.
There are a few ways to define them.

* If you want a variable to take integers ranging from 0 to 10 (e.g. for the variable `repetitions`), use ``"repetitions" : range(10)``.
* If you want it to take floating point values from 0 to 1 with steps of 0.1, like in `percentage`, use ``"percentage" : np.arange(0, 1.01, 0.1)``. Note that I used 1.01 because by definition of the function `arange()` the last point is excluded.
* If you want any other combination of values you can use Python lists, defined as in ``"variable": [1, 4, 213]``.

When done, continue with :ref:`run`.
