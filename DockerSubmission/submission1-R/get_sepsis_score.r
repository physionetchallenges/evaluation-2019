#!/usr/bin/Rscript

get_sepsis_score = function(data){
    x_mean = c(
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777)
    x_std = c(
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997)
    c_mean = c(60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551)
    c_std = c(16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367)

    m = dim(data)[1]
    n = dim(data)[2]
    x = data[1:m, 1:34]
    c = data[1:m, 35:40]

    x_norm = matrix(, m, 34)
    c_norm = matrix(, m, 6)
    for (i in 1:m){
        x_norm[i, 1:34] = (x[i, 1:34] - x_mean) / x_std
        c_norm[i, 1:6] = (c[i, 1:6] - c_mean) / c_std
    }

    x_norm[is.nan(x_norm)] = 0
    c_norm[is.nan(c_norm)] = 0

    beta = c(
        0.1806,  0.0249, 0.2120,  -0.0495, 0.0084,
        -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
        0.7476,  0.0323, 0.0305,  -0.0251, 0.0330,
        0.1424,  0.0324, -0.1450, -0.0594, 0.0085,
        -0.0501, 0.0265, 0.0794,  -0.0107, 0.0225,
        0.0040,  0.0799, -0.0287, 0.0531,  -0.0728,
        0.0243,  0.1017, 0.0662,  -0.0074, 0.0281,
        0.0078,  0.0593, -0.2046, -0.0167, 0.1239)
    rho = 7.8521
    nu = 1.0389

    xstar = cbind(x_norm, c_norm)
    exp_bx = exp(xstar %*% beta)
    l_exp_bx = (4.0 / rho) ** nu * exp_bx

    scores = 1 - exp(-l_exp_bx)
    labels = (scores > 0.45)
    results = cbind(scores, labels)
    return(results)
    }

read_challenge_data = function(input_file){
    data = data.matrix(read.csv(input_file, sep='|'))
    column_names = colnames(data)

    # ignore SepsisLabel column if present
    m = dim(data)[1]
    n = dim(data)[2]
    if (column_names[n] == 'SepsisLabel'){
        data = data[1:m, 1:n-1]
    }

    return(data)
    }

# load library and arguments
args = commandArgs(trailingOnly=TRUE)

# get input filenames
tmp_input_dir = 'tmp_inputs'
unzip(args[1], files = NULL, exdir = tmp_input_dir)
input_files = sort(list.files(path = 'tmp_inputs', recursive = TRUE))

# make temporary output directory
tmp_output_dir = 'tmp_outputs'
dir.create(tmp_output_dir)

# generate scores
n = length(input_files)
output_files = list()

for (i in 1:n){
    # read data
    input_file = file.path(tmp_input_dir, input_files[i])
    data = read_challenge_data(input_file)

    # make predictions
    results = get_sepsis_score(data)
    colnames(results) = c('PredictedProbability', 'PredictedLabel')

    # write results
    file_path = unlist(strsplit(input_file, .Platform$file.sep))
    file_name = file_path[length(file_path)]
    output_file = file.path(tmp_output_dir, file_name)
    write.table(results, file = output_file, sep = '|', quote=FALSE, row.names = FALSE)
    output_files[i] = output_file
}

# perform clean-up
zip(args[2], files = unlist(output_files))
unlink(tmp_input_dir, recursive=TRUE)
unlink(tmp_output_dir, recursive=TRUE)