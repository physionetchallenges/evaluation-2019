#!/bin/bash
#
# file: next.sh
#
# This bash script analyzes the record named in its command-line
# argument ($1), and writes per-hour classification to the file
# "$1.out".  This script is run once for each record in the Challenge
# test set.  The input data file ($1.psv) will be located in the
# current working directory.
#
# The output file must contain one line for each hour of the input
# record, and two columns separated by commas.  The first column is a
# value indicating the probability of sepsis and the second column is
# a binary classification (0 = no sepsis, 1 = sepsis).

set -e
set -o pipefail

RECORD=$1

# Example (Python)
./get_sepsis_score.py "$RECORD"
