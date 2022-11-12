% MACHINE LEARNING PROJECT: KNN EMOTION CLASSIFICATION
clear all
clc
addpath(genpath(pwd));
disp(pwd)
CSV_PATH = 'Data_path.csv';

% loading csv with audio path of all files
csv_file = readtable(CSV_PATH);

%LENGTH FOR EACH CLASS*****************************************************
%disp('computing length for each class (in seconds)');
%{
for i = 0:5
    class_len = 0;
    for j = 1:height(csv_file)
        if j == 18076
            continue
        end
        if j == 5041
            continue
        end
        line = csv_file(j, 1:4);
        path = string(line.path);
        label = string(line.labels);
        % leaving out gender
        %if string(line.gender) == 'male'
        %    continue
        %end
        switch i
            case 0
                if label == 'angry'
                    class_len = class_len + file_length(path);
                end
            case 1
                if label == 'neutral'
                    class_len = class_len + file_length(path);
                end
            case 2
                if label == 'sad'
                    class_len = class_len + file_length(path);
                end
            case 3
                if label == 'happy'
                    class_len = class_len + file_length(path);
                end
            case 4
                if label == 'fear'
                    class_len = class_len + file_length(path);
                end
            case 5
                if label == 'disgust'
                    class_len = class_len + file_length(path);
                end
        end
    end
    switch i
        case 0
            disp(['angry' string(class_len)]);
        case 1
            disp(['neutral' string(class_len)]);
        case 2
            disp(['sad' string(class_len)]);
        case 3
            disp(['happy' string(class_len)]);
        case 4
            disp(['fear' string(class_len)]);
        case 5
            disp(['disgust' string(class_len)]);
    end
end
%}
% x gender 
%{
for i = 0:1
    class_len = 0;
    for j = 1:height(csv_file)
        if j == 18076
            continue
        end
        if j == 5041
            continue
        end
        line = csv_file(j, 1:4);
        path = string(line.path);
        label = string(line.gender);
        switch i
            case 0
                if label == 'female'
                    class_len = class_len + file_length(path);
                end
            case 1
                if label == 'male'
                    class_len = class_len + file_length(path);
                end
        end
    end
    switch i
        case 0
            disp(['female' string(class_len)]);
        case 1
            disp(['male' string(class_len)]);
    end
end
%}
% KNN CLASSIFICATION EMOTIONS**********************************************
disp('loading classes data');
for i = 0:5
    list_class_path = [];
    for j = 1:height(csv_file)
        if j == 18076
            continue
        end
        if j == 5041
            continue
        end
        line = csv_file(j, 1:4);
        % leaving out female
        %if string(line.gender) == 'female'
        %    continue
        %end
        path = string(line.path);
        label = string(line.labels);
        switch i
            case 0
                if label == 'angry'
                    list_class_path = [list_class_path path];
                end
            case 1
                if label == 'neutral'
                    list_class_path = [list_class_path path];
                end
            case 2
                if label == 'sad'
                    list_class_path = [list_class_path path];
                end
            case 3
                if label == 'happy'
                    list_class_path = [list_class_path path];
                end
            case 4
                if label == 'fear'
                    list_class_path = [list_class_path path];
                end
            case 5
                if label == 'disgust'
                    list_class_path = [list_class_path path];
                end
        end
    end
    label = '';
    switch i
        case 0
            label = 'angry';
        case 1
            label = 'neutral';
        case 2
            label = 'sad';
        case 3
            label = 'happy';
        case 4
            label = 'fear';
        case 5
            label = 'disgust';
    end
    disp(['loading class:' string(i)]);
    kNN_model_add_class('K_NN.mat', label, list_class_path, ...
                {'mean', 'std'}, 0.200, 0.100, 3.0, 1.5);
end
%0.200, 0.100, 3.0, 1.5 -> 1 -> best

% KNN CLASSIFICATION GENDER************************************************
%{
list_class_path_male = [];
list_class_path_female = [];
for j = 1:height(csv_file)
    if j == 18076
        continue
    end
    if j == 5041
        continue
    end
    line = csv_file(j, 1:4);
    if string(line.labels) == 'surprise'
        continue
    end
    path = string(line.path);
    label = string(line.gender);
    if label == 'male'
        list_class_path_male = [list_class_path_male path];
        
    end
    if label == 'female'
        list_class_path_female = [list_class_path_female path];
    end
end
kNN_model_add_class('K_NN_gender.mat', 'male', list_class_path_male, ...
            {'mean', 'std'}, 0.200, 0.100, 3.0, 1.5);
kNN_model_add_class('K_NN_gender.mat', 'female', list_class_path_female, ...
            {'mean', 'std'}, 0.200, 0.100, 3.0, 1.5);

%}

disp('done.');
