SPAUN README
============

To run spaun, follow these instructions:
1. Install nengo 1403 as per the readme.txt file
2. Run nengo 1403
3. Within nengo, press CTRL+P to bring up the scripting console,
   and type "run run_spaun_win.py" (if you are on a windows-based 
   system) or "run run_spaun_lin.py" (if you are on a linux/unix-based
   system).
4. The log files that are generated can be found in spaun/out_data
   located within this directory.

Notes:
------
> This model requires a machine with at least 24GB of RAM to run.
  Estimated run times for a quad-core 2.5GHz are 3 hours per 1 second
  of simulation time.
> See the run_spaun.py file in the spaun directory for experiement
  options.
> Model parameters can be altered in the conf.py file found in the 
  spaun directory. 
> To load the model with a different number of dimensions than the 
  default (default = 512 dimensions), alter the line "num_dim = 512"
  in the conf.py file. Setting the number to 4 enables the model to 
  be loaded on a 2GB machine.