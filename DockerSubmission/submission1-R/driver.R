#!/usr/bin/Rscript

source("get_sepsis_score.R")

load_challenge_data = function(file){
    data = data.matrix(read.csv(file, sep='|'))
    column_names = colnames(data)

    # ignore SepsisLabel column if present
    if (column_names[ncol(data)] == 'SepsisLabel'){
        data = data[, 1:ncol(data)-1]
    }

    return(data)
}

save_challenge_predictions = function(file, predictions){
    colnames(predictions) = c('PredictedProbability', 'PredictedLabel')
    write.table(predictions, file = file, sep = '|', quote = FALSE, row.names = FALSE)
}

# Parse arguments.
args = commandArgs(trailingOnly=TRUE)
if (length(args) != 2 && length(args) != 3){
    stop('Include the input and output directories as arguments, e.g., Rscript driver.r input output.')
}

input_directory = args[1]
output_directory = args[2]

if (length(args) == 2 || args[3] == '0' || args[3] == 'F' || args[3] == 'FALSE'){
    enforce_causality = FALSE
} else{
    enforce_causality = TRUE
}

# Find files.
files = c()
for (f in list.files(input_directory)){
    if (file.exists(file.path(input_directory, f)) && nchar(f) >= 3 && substr(f, 1, 1) != '.' && substr(f, nchar(f)-2, nchar(f)) == 'psv'){
        files = c(files, f)
    }
}

if (!dir.exists(output_directory)){
    dir.create(output_directory)
}

# Iterate over files.
for (file in files){
    # Load data.
    input_file = file.path(input_directory, file)
    data = load_challenge_data(input_file)

    # Make predictions.
    if (!enforce_causality){
        predictions = get_sepsis_score(data)
    } else{
        num_rows = nrow(data)
        predictions = matrix(, num_rows, 2)
        for (t in 1 : num_rows){
            current_data = matrix(data[1 : t, ], t, ncol(data))
            current_predictions = get_sepsis_score(current_data)
            predictions[t, ] = current_predictions[t, ]
        }
    }

    # Save results.
    output_file = file.path(output_directory, file)
    save_challenge_predictions(output_file, predictions)
}
