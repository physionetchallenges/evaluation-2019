function driver(input_directory, output_directory)
    % Find files.
    files = {};
    for f = dir(input_directory)'
        if exist(fullfile(input_directory, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'psv')
            files{end + 1} = f.name;
        end
    end

    if ~exist(output_directory, 'dir')
        mkdir(output_directory)
    end

    % Load model.
    model = load_sepsis_model();

    % Iterate over files.
    num_files = length(files);
    for i = 1:num_files
        % Load data.
        input_file = fullfile(input_directory, files{i});
        data = load_challenge_data(input_file);

        % Make predictions.
        num_rows = size(data, 1);
        scores = zeros(num_rows, 1);
        labels = zeros(num_rows, 1);
        for t = 1:num_rows
            current_data = data(1:t, :);
            [current_score, current_label] = get_sepsis_score(current_data, model);
            scores(t) = current_score;
            labels(t) = current_label;
        end

        % Save results.
        output_file = fullfile(output_directory, files{i});
        save_challenge_predictions(output_file, scores, labels);
    end
end

function data = load_challenge_data(file)
    f = fopen(file, 'rt');
    try
        l = fgetl(f);
        column_names = strsplit(l, '|');
        data = dlmread(file, '|', 1, 0);
    catch ex
        fclose(f);
        rethrow(ex);
    end
    fclose(f);

    % Ignore SepsisLabel column if present.
    if strcmp(column_names(end), 'SepsisLabel')
        column_names = column_names(1 : end-1);
        data = data(:, 1 : end-1);
    end
end

function save_challenge_predictions(file, scores, labels)
    fid = fopen(file, 'wt');
    fprintf(fid, 'PredictedProbability|PredictedLabel\n');
    fclose(fid);
    dlmwrite(file, [scores labels], 'delimiter', '|', '-append');
end
