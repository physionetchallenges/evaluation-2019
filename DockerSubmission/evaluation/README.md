# Instructions

The Python script `evaluate_sepsis_score.py` evaluates predictions on the input data.

To run this script, install the NumPy Python package and run

        python evaluate_sepsis_score.py labels predictions scores.psv

where `labels` is a directory containing files with labels, such as the training data on the PhysioNet website; `predictions` is a directory containing files with predictions produced by your algorithm; and `scores.psv` (optional) is a collection of scores for the predictions (described on the PhysioNet website). We use the utility score for the Challenge.
