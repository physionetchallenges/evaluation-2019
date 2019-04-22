# Instructions

The Python script `get_sepsis_score.py` makes sepsis predictions on clinical time-series data.  The Python script `driver.py` is a helper script that loads the data, calls `get_sepsis_score.py`, and saves the predictions.

Please add your prediction code to the function `get_sepsis_score` in the `get_sepsis_score.py` script.  Please do *not* change the `driver.py` script or the format of the inputs and outputs for the `get_sepsis_score` function -- or we will be unable to evaluate your submission.

You can run your prediction code on a patient cohort by running

        python driver.py input_directory output_directory

where `input_directory` contains input data files and `output_directory` contains output prediction files.  The input files are provided in a training database available on the PhysioNet website, and the format for the output files is described on the PhysioNet website.
