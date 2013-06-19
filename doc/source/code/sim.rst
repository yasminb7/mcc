.. _sim-parameters:

sim Package
===========

=================================== 	==============================================================================================================================================================================================================================================================
                                     	
**Preparation**                      	
`get`                                  	Name of the function that returns const in the right form
                                     	
**Generalities**                     	
`name`                                 	Name of folder including \%s for variables to be replaced according to ``factors``
`factors`                              	Which of the variables in const should be interpreted as a range of values to simulate?
`repetitions`                          	How many times this simulation will be repeated
`max_time`                             	Maximum simulation time
`dt`                                   	Timestep
`N_amoeboid`                           	Number of amoeboid agents
`N_mesenchymal`                        	Number of mesenchymal agents
`percentage`                           	Percentage of mesenchymal agents (of the total population)
                                     	
**Initial conditions**               	
`success_radius`                       	The cercle inside of which agents are considered successful has radius:
`initial_position`                     	The agents' initial positions follow a Gaussian distribution around this point
`initial_position_stray`               	The standard deviation of said distribution
                                     	
**Parameters that apply to agents**  	
`q`                                    	Energy intake
`delta`                                	Energy dissipation
`mass`                                 	Mass of an agent
`gamma_a`                              	Stokes' friction = gamma ** v (for amoeboids)
`gamma_m`                              	Stokes' friction = gamma ** v (for mesenchymals)
`eta`                                  	Eta is a constant characterizing propulsion
`radius`                               	Radius of agents
`orientationperiod`                    	Each ? time units the agents reorient themselves according to the gradient
`periodsigma`                          	Describes the variation in the above quantity
`orientationdelay`                     	It takes them #? time units to do so
`delaysigma`                           	Describes the variation in the above quantity
`compass_noise_a`                      	When the agent reorients, the new direction is the one given by the concentration gradient + gaussian noise with sigma given by this value
`compass_noise_m`                      	When the agent reorients, the new direction is the one given by the concentration gradient + gaussian noise with sigma given by this value
                                     	
**Forces and interactions**          	
`r`                                    	Factor of the stochastic force
`enable_interaction`                   	Enable interaction, True or False
`interaction_radius`                   	For distances smaller than this, agents experience repulsion
`alignment_radius`                     	For distances between interaction_radius and this, there is alignment
`repulsion_coupling`                   	Constant governing repulsion
`w`                                    	The weight that describes the alignment weight (but alignment happens at each step, so it is multiplied by dt later to compensate)
                                     	
**Environment**                      	
`fieldlimits`                          	The coordinates that are considered to be inside of the field (x_min, x_max, y_min, y_max)
`border`                               	The thickness of the field's border
`nodegradationlimit`                   	The distance of the limits of the field at which the agents aren't allowed to degrade anymore
`gradientcenter`                       	The center of the gradient field (x, y)
`maze`                                 	The path to the image that describes the maze
`wall`                                 	The float value that a wall should have (should remain 1.0)
`wall_limit`                           	A limit value of what is still considered a wall (don't touch this unless you must)
                                     	
**Mesenchymals**                     	
`eatshape`                             	``eatshape`` is a stupid name for the file that describes how surrounding pixels are affected by degradation
`degradation_radius`                   	If an agent has been degrading at some point `p`, it will not degrade until it's at a distance of ``degradation_radius`` away from `p`. There is currently a problem with this (see energy graphs for individual agents), but it doesn't matter at this stage.
`zeta`                                 	Describes how costly degradation is in terms of energy.
`safety_factor`                        	The safety factor describes what the sides of the rectangle containing the degradation imprints are multiplied with when updating the gradient.
`aura`                                 	Unused
`y`                                    	Unused.
                                     	
**After the simulation**             	
`numframes`                            	Unused
`fps`                                  	How many frames per unit of elapsed simulation time should be created for the video.
`create_path_plot`                     	Currently unused. Whether to create a path plot (trajectories) **immediately** after the simulation or not.
`create_video_directly`                	Currently unused. Whether to create a video **immediately** after the simulation or not.
`simulations_with_complete_dataset`    	In the case of the parameter `repetitions` being larger than 1, how many repetitions should save the full dataset after being run.
`handle_repetitions_with`              	What to do when calculating values (from finalstats) from many repetitions: mean or median?
=================================== 	==============================================================================================================================================================================================================================================================




:mod:`sim` Package
------------------

.. automodule:: sim
    :members:
    :undoc-members:
    :show-inheritance:


:mod:`prototype` Module
-----------------------

.. automodule:: sim.prototype
    :members:
    :undoc-members:
    :show-inheritance:
