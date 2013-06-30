Implementation
==============

This chapter is intended to help you understand how the model described in the thesis has been implemented in the code.

What happens where
------------------

The actual calculations of the forces, velocities, positions etc. is done in the function :py:meth:`.Simulation.run`. That function is where the algorithm from pages 19 and 20 of the thesis is implemented.

**Almost every possible change you intend to make to the model, will have to be implemented in that function.**
Only a few specific calculations are done outside of the main loop; these are described in :ref:`helpers`.

In the method :py:meth:`.Simulation.run`, basically three things happen:

* The simulation is set up. Among others:

  * variables are set up for later use
  * an object of the class :py:class:`.Maze` is created, loading the image representing the maze from the `resources/` folder
  * agents are initialised

* Then the main loop is executed. In order to understand what it does, as a first reading the algorithm described in my thesis should be helpful. The code itself has lots of comments, so I kindly ask you to read these. A link to the code is included in the documentation of :py:meth:`.Simulation.run`.

* After the main loop, the dataset is stored and some details are taken care of.

Pointers to crucial pieces of code
----------------------------------

As already mentioned, you can find the source code (at the time this documentation was written) using a link in :py:meth:`.Simulation.run`.


* The **interaction between agents** is implemented in lines 193 to 252.
* Lines 254 to 287 contain the evolution of the 2 steps of **chemotaxis**.
* In lines 290 to 302 the **forces** acting on all agents are calculated.
* Lines 304 to 355 are responsible for checking which agents are in contact with the ECM and which should degrade the ECM, and for evolving their velocities and positions accordingly.
* The **degradation of the ECM** happens in lines 357 to 390.
* In lines 392 to 398 the energy values of the agents are updated.


How agents are described
------------------------

The agents are *not* represented as objects of a class, instead their variables can be accessed through the :py:class:`.Dataset` object. For example, at timestep `i` agent `k` has the energy stored in

>>> self.dsA.energies[i,k]

Agents are described by the values that are stored in NumPy-arrays in the object of the class :py:class:`.Dataset`. These arrays are:

* times
* types
* positions
* velocities
* energies
* states
* statechanges
* periods
* delays
* eating
* direction_angles

These arrays are defined by :py:attr:`.Dataset.ARRAYS` and further described in :ref:`dataset-arrays`.

During and after a simulation, all the everything you could want to know about an agent is stored in these arrays and can be accessed as described above in the section on the main loop. For example

>>> self.dsA.positions[i,k]

is the position at timestep **i** of agent **k** (remember that indexing starts at zero). The above statement will give you array of shape ``(2,)`` because positions are two-dimensional values. If you wanted to access the `x` value of the velocity of agent 24 at timestep 113, you would write

>>> self.dsA.velocities[113, 24, 0]


..	_helpers:

Helper functions
----------------

The main loop does rely on some outside functions for specific tasks.
These functions are:

* The method :py:meth:`getGradients` from the class :py:class:`.Maze`. This method in turn will call :py:meth:`.getGradientsCpp` or :py:meth:`.getGradientsPython` depending on whether or not you activated support for C in :py:mod:`.constants`.
* :py:meth:`.Maze.degrade()` is responsible for modifing the array representing the maze in the positions where mesenchymals are currently degrading it.
* :py:func:`.statutils.getDistancesSq` returns the squared distances of all the agents from the goal.


