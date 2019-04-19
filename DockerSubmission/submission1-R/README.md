# Instructions

The R script `get_sepsis_score.R` makes sepsis predictions on clinical time-series data.  The R script `driver.R` is a helper script that loads the data, calls `get_sepsis_score.R`, and saves the predictions.

Please add your prediction code to the function `get_sepsis_score` in the `get_sepsis_score.R` script.  Please do *not* change the `driver.R` script or the format of the inputs and outputs for the `get_sepsis_score` function -- or we will be unable to evaluate your submission.

You can run your prediction code on a patient cohort by running

        Rscript driver.R input_directory output_directory

where `input_directory` contains input data files and `output_directory` contains output prediction files.  The input files are provided in a training database available on the PhysioNet website, and the format for the output files is described on the PhysioNet website.
