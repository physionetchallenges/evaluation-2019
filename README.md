# Evaluation code for the PhysioNet/CinC Challenge 2019

## Contents

The MATLAB script `evaluate_sepsis_score.m` and Python script `evaluate_sepsis_score.py` evaluate predictions from your algorithm using a utility-based evaluation metric that we designed for the PhysioNet/CinC Challenge 2019.  These scripts produce the same results.

## Running

### MATLAB

You can run the MATLAB evaluation code by installing by running

    evaluate_sepsis_score(labels, predictions, 'scores.psv')

where `labels` is a directory containing files with labels, such as the training database on the PhysioNet webpage; `predictions` is a directory containing files with predictions produced by your algorithm; and `scores.psv` (optional) is a collection of scores for the predictions (described on the PhysioNet website).

### Python

You can run the Python evaluation code by installing the NumPy Python package and running

    python evaluate_sepsis_score.py labels predictions scores.psv

where `labels` is a directory containing files with labels, such as the training database on the PhysioNet webpage; `predictions` is a directory containing files with predictions produced by your algorithm; and `scores.psv` (optional) is a collection of scores for the predictions (described on the PhysioNet website).

For PhysioNet/CinC 2019, we use the utility score, which is the last (fifth) score in the output of this function.

## Example prediction code

This repository contains evaluation code for the PhysioNet/CinC Challenge 2019.  **Looking for Julia, MATLAB, Python, or R example prediction code?**  See the repositories in <https://github.com/physionetchallenges>.
