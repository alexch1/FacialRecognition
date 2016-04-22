clear;
fprintf('           ---------Q0_preprocess--------\n');

training_folders_path = 'face_database/training_set';
testing_folders_path = 'face_database/testing_set';
training_forders = dir(training_folders_path);
testing_forders = dir(testing_folders_path);
training_struct = struct('training_feature_vector',{}, 'training_class_ID',{},'training_file_name',{},'type',{});
testing_struct = struct('testing_feature_vector',{},'testing_class_ID',{},'testing_file_name',{},'type',{});
wholedata_struct = struct('feature_vector',{}, 'class_ID',{},'file_name',{},'type',{});

training_file_name = '   ';
testing_file_name = '   ';
training_data_index = 1;
testing_data_index = 1;
training_class_ID = 0;
testing_class_ID = 0;

%train data processing
fprintf('           Training images analysing...\n');
for training_path_index = 1:length(training_forders)
    training_folder_name = training_forders(training_path_index).name; 
    temppath=fullfile(training_folders_path,training_folder_name);
    training_forder_list = dir(temppath);
    count=1;
    for training_file_index = 1:length(training_forder_list)
        if (training_forder_list(training_file_index).isdir == 0 && strcmp(training_forder_list(training_file_index).name(end-2:end),'png'))
            if strcmp(training_file_name(1:3),training_forder_list(training_file_index).name(1:3)) == 0
               training_class_ID = training_class_ID + 1;
            end
            training_file_name = strcat(training_forder_list(training_file_index).name(1:3),'-',int2str(count));
            training_feature_vector = imread([temppath '/' training_forder_list(training_file_index).name]);
            training_struct(training_data_index) = struct('training_feature_vector', im2double(training_feature_vector), 'training_class_ID', training_class_ID, 'training_file_name', training_file_name, 'type', true);
            wholedata_struct(training_data_index) = struct('feature_vector', im2double(training_feature_vector), 'class_ID', training_class_ID, 'file_name', strcat(training_file_name,'-train'), 'type', true);
            training_data_index = training_data_index + 1;
            count=count+1;
        end
    end
end

dimensions = size(training_struct(1).training_feature_vector);

%test data processing
fprintf('           Testing images analysing...\n');
for testing_path_index = 1:length(testing_forders)
    testing_folder_name = testing_forders(testing_path_index).name; 
    temppath=fullfile(testing_folders_path,testing_folder_name);
    testing_forder_list = dir(temppath);
    count=1;
    for testing_file_index = 1:length(testing_forder_list)
        if (testing_forder_list(testing_file_index).isdir == 0 && strcmp(testing_forder_list(testing_file_index).name(end-2:end),'png'))
            if strcmp(testing_file_name(1:3),testing_forder_list(testing_file_index).name(1:3)) == 0
               testing_class_ID = testing_class_ID + 1;
            end
            testing_file_name = strcat(testing_forder_list(testing_file_index).name(1:3),'-',int2str(count));
            testing_feature_vector = imread([temppath '/' testing_forder_list(testing_file_index).name]);
            testing_struct(testing_data_index) = struct('testing_feature_vector', im2double(testing_feature_vector), 'testing_class_ID', testing_class_ID, 'testing_file_name', testing_file_name, 'type', false);
            wholedata_struct(testing_data_index+length(training_struct)) = struct('feature_vector', im2double(testing_feature_vector), 'class_ID', testing_class_ID, 'file_name', strcat(testing_file_name,'-test'), 'type', false);
            testing_data_index = testing_data_index + 1;
            count=count+1;
        end
    end
end
fprintf('           Saving "preprocess.mat"...\n');
save 'preprocess.mat';
fprintf('           --------------DONE!-----------\n\n');