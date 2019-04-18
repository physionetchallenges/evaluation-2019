#!/usr/bin/env python3

import numpy as np, os, sys
from get_sepsis_score import get_sepsis_score

def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # ignore SepsisLabel column if present
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        data = data[:, :-1]

    return data

def save_challenge_predictions(file, scores, labels):
    with open(file, 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        for (s, l) in zip(scores, labels):
            f.write('%g|%d\n' % (s, l))

if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    if len(sys.argv) == 3 or sys.argv[0] == '0' or sys.argv[0] == 'False':
        enforce_causality = False
    else:
        enforce_causality = True

    # Find files.
    files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)

    # Iterate over files.
    for file in files:
        # Load data.
        input_file = os.path.join(input_directory, file)
        data = load_challenge_data(input_file)

        # Make predictions.
        if not enforce_causality:
            scores, labels = get_sepsis_score(data)
        else:
            num_records = len(data)
            scores = np.zeros(num_records)
            labels = np.zeros(num_records)
            for t in range(num_records):
                current_data = data[:t+1]
                current_scores, current_labels = get_sepsis_score(current_data)
                scores[t] = current_scores[t]
                labels[t] = current_labels[t]

        # Save results.
        output_file = os.path.join(output_directory, file)
        save_challenge_predictions(output_file, scores, labels)
