# SnowstakeVision
NOTE: This project is still in progress and is not ready for use.

This model identifies snowstakes out of a forest background, and then deduces from an 150cm stake height the depth of snowpack in a given picture

The origin of this algorithm is based on snowstakes used by the Adaptive Silviculture for Climate Change study in thickly-forested Northeast Vermont. The model seeks to identify and isolate the snowstake, and then count the alternating red/ black bands from the top of the stake to deduce the height of snowpack in a given area. Such is done using Python's computer vision libraries, and other functions. The final product is an excel sheet that lists the names of files, location, time, and associated snow depths.

Such leverages computer vision libraries such as TkInter, and CV2, and hosts a user-friendly GUI for enhanced compatability.

