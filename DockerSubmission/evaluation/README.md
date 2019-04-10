# Instructions

The Python script `evaluate_sepsis_score.py` evaluates predictions on the input data.

To run this script, install the NumPy Python package and run

        python evaluate_sepsis_score.py labels.tar predictions.tar scores.psv

which takes `labels.tar` (a tar archive file of the data/labels files, which are available on the PhysioNet website) and `predictions.tar` (a tar archive file of the prediction files, which are described on the PhysioNet website) as input and returns `scores.psv` as output.
