% This file contains functions for computing scores for the 2019 PhysioNet/CinC
% challenge.
%
% Written by M. Reyna on 1 February 2019.  Last updated on 2 February 2019.
%
% The compute_utility_2019 function computes a normalized utility score for a
% cohort of patients.
%
% Inputs:
%   'labels_directory' is a directory of pipe-delimited text files containing a
%   binary vector of labels indicating whether a patient is not septic (0) or
%   septic (1).
%
%   'predictions_directory' is a directory of pipe-delimited text files, where
%   the first column of the file gives the  predicted probability that the
%   patient is septic at each time, and the second column of the file is a
%   binarized version of this vector. Note that there must be a prediction for
%   every label.
%
% Output:
%   'output_file' is a pipe-delimited text file (optional) that gives AUROC,
%   AUPRC, accuracy, F-measure, and utility scores for a cohort of patients.
%
% Example:
%
%   >> compute_scores_2019('labels', 'results', 'output.dat')

function compute_scores_2019(label_directory, prediction_directory, output_file)

% Set parameters.
label_header       = 'SepsisLabel';
prediction_header  = 'PredictedLabel';
probability_header = 'PredictedProbability';

dt_early   = -12;
dt_optimal = -6;
dt_late    = 3;

max_u_tp = 1;
min_u_fn = -2;
u_fp     = -0.05;
u_tn     = 0;

% Find label and prediction files.
files = dir(fullfile(label_directory, '*.dat'));
num_files = length(files);

label_files = cell(1, num_files);
for k = 1 : num_files
    label_files{k} = char(files(k).name);
end

files = dir(fullfile(prediction_directory, '*.dat'));
num_files = length(files);

prediction_files = cell(1, num_files);
for k = 1 : num_files
    prediction_files{k} = char(files(k).name);
end

if length(label_files) ~= length(prediction_files)
    error('Numbers of labels and predictions must be the same.');
end

% Load labels and predictions.
num_files            = length(label_files);
cohort_labels        = cell(1, num_files);
cohort_predictions   = cell(1, num_files);
cohort_probabilities = cell(1, num_files);

for k = 1 : num_files
    labels        = load_column(fullfile(label_directory, label_files{k}), label_header);
    predictions   = load_column(fullfile(prediction_directory, prediction_files{k}), prediction_header);
    probabilities = load_column(fullfile(prediction_directory, prediction_files{k}), probability_header);

    % Check labels and predictions for errors.
    if ~(length(labels) == length(predictions) || length(predictions) == length(probabilities))
        error('Numbers of labels and predictions must be the same.');
    end

    num_records = length(labels);

    for i = 1 : num_records
        if ~(labels(i) == 0 || labels(i) == 1)
            error('Labels must satisfy label == 0 or label == 1.');
        end

        if ~(predictions(i) == 0 || predictions(i) == 1)
            error('Predictions must satisfy prediction == 0 or prediction == 1.');
        end

        if ~(probabilities(i) >= 0 || probabilities(i) <= 1)
            error('Probabilities must satisfy 0 <= probability <= 1.');
        end
    end

    min_probability_positive = min(probabilities(predictions(:) == 1));
    max_probability_negative = max(probabilities(predictions(:) == 0));

    if min_probability_positive <= max_probability_negative
        error(['Predictions are inconsistent with probabilities, i.e.,'...
               'a positive prediction has a lower (or equal) probability than a negative prediction.']);
    end

    % Record labels and predictions.
    cohort_labels{k}        = labels;
    cohort_predictions{k}   = predictions;
    cohort_probabilities{k} = probabilities;
end

% Compute AUC, accuracy, and F-measure.
labels = [];
predictions = [];
probabilities = [];
for k = 1 : num_files
    labels        = [labels; cohort_labels{k}];
    predictions   = [predictions; cohort_predictions{k}];
    probabilities = [probabilities; cohort_probabilities{k}];
end

[auroc, auprc]        = compute_auc(labels, probabilities);
[accuracy, f_measure] = compute_accuracy_f_measure(labels, predictions);

% Compute utility.
observed_utilities = zeros(1, num_files);
best_utilities     = zeros(1, num_files);
worst_utilities    = zeros(1, num_files);
inaction_utilities = zeros(1, num_files);

for k = 1 : num_files
    labels = cohort_labels{k};
    num_records          = length(labels);
    observed_predictions = cohort_predictions{k};
    best_predictions     = zeros(1, num_records);
    worst_predictions    = zeros(1, num_records);
    inaction_predictions = zeros(1, num_records);

    if any(labels)
        t_sepsis = find(labels == 1, 1);
        best_predictions(max(1, t_sepsis + dt_early) : min(t_sepsis + dt_late, num_records)) = 1;
    end
    worst_predictions = 1 - best_predictions;

    observed_utilities(k) = compute_prediction_utility(labels, observed_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn);
    best_utilities(k)     = compute_prediction_utility(labels, best_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn);
    worst_utilities(k)    = compute_prediction_utility(labels, worst_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn);
    inaction_utilities(k) = compute_prediction_utility(labels, inaction_predictions, dt_early, dt_optimal, dt_late, max_u_tp, min_u_fn, u_fp, u_tn);
end

unnormalized_observed_utility = sum(observed_utilities);
unnormalized_best_utility     = sum(best_utilities);
unnormalized_worst_utility    = sum(worst_utilities);
unnormalized_inaction_utility = sum(inaction_utilities);

if ~(unnormalized_worst_utility <= unnormalized_best_utility && unnormalized_inaction_utility <= unnormalized_best_utility)
    error('Optimal utility must be higher than inaction utility.');
end

normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility);

% Output results.
output_string = sprintf('AUROC|AUPRC|Accuracy|F-measure|Utility\n%f|%f|%f|%f|%f',...
                        auroc, auprc, accuracy, f_measure, normalized_observed_utility);
switch nargin
    case 2
        disp(output_string)
    case 3
        fid = fopen(output_file, 'wt');
        fprintf(fid, output_string);
        fclose(fid);
end
end

% The load_column function loads a column from a table.
%
% Inputs:
%   'filename' is a string containing a filename.
%
%   'header' is a string containing a header.
%
% Outputs:
%   'column' is a vector containing a column from the file with the given
%   header.
%
% Example:
%
%   Omitted.

function column = load_column(filename, header)
tbl  = readtable(filename);
if size(tbl)
    column = table2array(tbl(:, header));
else
    column = zeros(1, 0);
end
end

% The compute_auc function computes AUROC and AUPRC as well as other summary
% statistics (TP, FP, FN, TN, TPR, TNR, PPV, NPV, etc.) that can be exposed
% from this function.
%
% Inputs:
%   'labels' is a binary vector, where labels(i) == 0 if the patient is not
%   labeled as septic at time i and labels(i) == 1 if the patient is labeled as
%   septic at time i.
%
%   'predictions' is a probability vector, where predictions(i) gives the
%   predicted probability that the patient is septic at time i.  Note that there
%   must be a prediction for every label, i.e, length(labels) ==
%   length(predictions).
%
% Outputs:
%   'auroc' is a scalar that gives the AUROC of the classifier using its
%   predicted probabilities, where specificity is interpolated for intermediate
%   sensitivity values.
%
%   'auprc' is a scalar that gives the AUPRC of the classifier using its
%   predicted probabilities, where precision is a piecewise constant function of
%   recall.
%
% Example:
%
%   >> labels = [0; 0; 0; 0; 1; 1];
%   >> predictions = [0.3; 0.4; 0.6; 0.7; 0.8; 0.8];
%   >> [auroc, auprc] = compute_auc(labels, predictions)
%   auroc = 1
%   auprc = 1

function [auroc, auprc] = compute_auc(labels, predictions)
% Check inputs for errors.
if length(predictions) ~= length(labels)
    error('Numbers of predictions and labels must be the same.');
end

n = length(labels);
for i = 1 : n
    if ~(labels(i) == 0 || labels(i) == 1)
        error('Labels must satisfy label == 0 or label == 1.');
    end
end

for i = 1 : n
    if ~(predictions(i) >= 0 && predictions(i) <= 1)
        error('Predictions must satisfy 0 <= prediction <= 1.');
    end
end

% Find prediction thresholds.
thresholds = flipud(unique(predictions));

if thresholds(1) ~= 1
    thresholds = [1; thresholds];
end

if thresholds(end) ~= 0
    thresholds = [thresholds; 0];
end

m = length(thresholds);

% Populate contingency table across prediction thresholds.
tp = zeros(1, m);
fp = zeros(1, m);
fn = zeros(1, m);
tn = zeros(1, m);

% Find indices that sort predicted probabilities from largest to smallest.
[~, idx] = sort(predictions, 'descend');

i = 1;
for j = 1 : m
    % Initialize contingency table for j-th prediction threshold.
    if j == 1
        tp(j) = 0;
        fp(j) = 0;
        fn(j) = sum(labels(:) == 1);
        tn(j) = sum(labels(:) == 0);
    else
        tp(j) = tp(j - 1);
        fp(j) = fp(j - 1);
        fn(j) = fn(j - 1);
        tn(j) = tn(j - 1);
    end

    % Update contingency table for i-th largest prediction probability.
    while i <= n && predictions(idx(i)) >= thresholds(j)
        if labels(idx(i)) == 1
            tp(j) = tp(j) + 1;
            fn(j) = fn(j) - 1;
        else
            fp(j) = fp(j) + 1;
            tn(j) = tn(j) - 1;
        end
        i = i + 1;
    end
end

% Summarize contingency table.
tpr = zeros(1, m);
tnr = zeros(1, m);
ppv = zeros(1, m);
npv = zeros(1, m);

for j = 1 : m
    if tp(j) + fn(j) > 0
        tpr(j) = tp(j) / (tp(j) + fn(j));
    else
        tpr(j) = 1;
    end

    if fp(j) + tn(j) > 0
        tnr(j) = tn(j) / (fp(j) + tn(j));
    else
        tnr(j) = 1;
    end

    if tp(j) + fp(j) > 0
        ppv(j) = tp(j) / (tp(j) + fp(j));
    else
        ppv(j) = 1;
    end

    if fn(j) + tn(j) > 0
        npv(j) = tn(j) / (fn(j) + tn(j));
    else
        npv(j) = 1;
    end
end

% Compute AUROC as the area under a piecewise linear function of TPR /
% sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
% under a piecewise constant of TPR / recall (x-axis) and PPV / precision
% (y-axis).
auroc = 0;
auprc = 0;
for j = 1 : m - 1
    auroc = auroc + 0.5 * (tpr(j + 1) - tpr(j)) * (tnr(j + 1) + tnr(j));
    auprc = auprc + (tpr(j + 1) - tpr(j)) * ppv(j + 1);
end
end

% The compute_accuracy_f_measure function computes the accuracy and F-measure
% for a patient.
%
% Inputs:
%   'labels' is a binary vector, where labels(i) == 0 if the patient is not
%   labeled as septic at time i and labels(i) == 1 if the patient is labeled as
%   septic at time i.
%
%   'predictions' is a binary vector, where predictions(i) == 0 if the patient
%   is not predicted to be septic at time i and predictions(i) == 1 if the
%   patient is predicted to be septic at time i.  Note that there must be a
%   prediction for every label, i.e, length(labels) == length(predictions).
%
% Output:
%   'accuracy' is a scalar that gives the accuracy of the classifier using its
%   binarized predictions.
%
%   'f_measure' is a scalar that gives the F-measure of the classifier using its
%   binarized predictions.
%
% Example:
%   >> labels = [0; 0; 0; 0; 1; 1]
%   >> predictions = [0 0 1 1 1 1]
%   >> [accuracy, f_measure] = compute_accuracy_f_measure(labels, predictions)
%   accuracy = 0.66667
%   f_measure = 0.66667

function [accuracy, f_measure] = compute_accuracy_f_measure(labels, predictions)
% Check inputs for errors.
if length(predictions) ~= length(labels)
    error('Numbers of predictions and labels must be the same.');
end

n = length(labels);
for i = 1 : n
    if ~(labels(i) == 0 || labels(i) == 1)
        error('Labels must satisfy label == 0 or label == 1.');
    end
end

for i = 1 : n
    if ~(predictions(i) == 0 || predictions(i) == 1)
        error('Predictions must satisfy prediction == 0 or prediction == 1.');
    end
end

% Populate contingency table.
tp = 0;
fp = 0;
fn = 0;
tn = 0;

for i = 1 : n
    if labels(i) == 1 && predictions(i) == 1
        tp = tp + 1;
    elseif labels(i) == 1 && predictions(i) == 0
        fp = fp + 1;
    elseif labels(i) == 0 && predictions(i) == 1
        fn = fn + 1;
    elseif labels(i) == 0 && predictions(i) == 0
        tn = tn + 1;
    end
end

% Summarize contingency table.
if tp + fp + fn + tn > 0
    accuracy = (tp + tn) / (tp + fp + fn + tn);
else
    accuracy = 1;
end

if 2 * tp + fp + fn > 0
    f_measure = 2 * tp / (2 * tp + fp + fn);
else
    f_measure = 1;
end
end

% The compute_prediction_utility function computes the total time-dependent
% utility for a patient.
%
% Inputs:
%   'labels' is a binary vector, where labels(i) == 0 if the patient is not
%   labeled as septic at time i and labels(i) == 1 if the patient is labeled as
%   septic at time i.
%
%   'predictions' is a binary vector, where predictions(i) == 0 if the patient
%   is not predicted to be septic at time i and predictions(i) == 1 if the
%   patient is predicted to be septic at time i.  Note that there must be a
%   prediction for every label, i.e, length(labels) == length(predictions).
%
% Output:
%   'utility' is a scalar that gives the total time-dependent utility of the
%   classifier using its binarized predictions.
%
% Example:
%   >> labels = [0; 0; 0; 0; 1; 1]
%   >> predictions = [0 0 1 1 1 1]
%   >> utility = compute_prediction_utility(labels, predictions)
%   utility = 0.44444

function utility = compute_prediction_utility(labels, predictions, dt_early,dt_optimal,...
                                              dt_late, max_u_tp, min_u_fn, u_fp, u_tn)
% Define parameters for utility functions.
switch nargin
    case 2
        dt_early   = -12;
        dt_optimal = -6;
        dt_late    = 3;

        max_u_tp = 1;
        min_u_fn = -2;
        u_fp     = -0.05;
        u_tn     = 0;
end

% Check inputs for errors.
if length(predictions) ~= length(labels)
    error('Numbers of predictions and labels must be the same.');
end

n = length(labels);
for i = 1 : n
    if ~(labels(i) == 0 || labels(i) == 1)
        error('Labels must satisfy label == 0 or label == 1.');
    end
end

for i = 1 : n
    if ~(predictions(i) == 0 || predictions(i) == 1)
        error('Predictions must satisfy prediction == 0 or prediction == 1.');
    end
end

if dt_early >= dt_optimal
    error('The earliest beneficial time for predictions must be before the optimal time.')
end

if dt_optimal >= dt_late
    error('The optimal time for predictions must be before the latest beneficial time.')
end

% Does the patient eventually have sepsis?
if any(labels)
    is_septic = true;
    t_sepsis = find(labels == 1, 1);
else
    is_septic = false;
    t_sepsis = inf;
end

% Define slopes and intercept points for affine utility functions of the
% form u = m * t + b.
m_1 = max_u_tp / (dt_optimal - dt_early);
b_1 = -m_1 * dt_early;
m_2 = -max_u_tp / (dt_late - dt_optimal);
b_2 = -m_2 * dt_late;
m_3 = min_u_fn / (dt_late - dt_optimal);
b_3 = -m_3 * dt_optimal;

% Compare predicted and true conditions.
u = zeros(1, n);

for t = 1 : n
    if t <= t_sepsis + dt_late

        % TP
        if is_septic && predictions(t)
            if t <= t_sepsis + dt_optimal
                u(t) = max(m_1 * (t - t_sepsis) + b_1, u_fp);
            elseif t <= t_sepsis + dt_late
                u(t) = m_2 * (t - t_sepsis) + b_2;
            end

        % FN
        elseif is_septic && ~predictions(t)
            if t <= t_sepsis + dt_optimal
                u(t) = 0;
            elseif t <= t_sepsis + dt_late
                u(t) = m_3 * (t - t_sepsis) + b_3;
            end

        % FP
        elseif ~is_septic && predictions(t)

            u(t) = u_fp;

        % TN
        elseif ~is_septic && ~predictions(t)
            u(t) = u_tn;
        end

    end
end

% Find total utility for patient.
utility = sum(u);
end
