clear;
load 'preprocess.mat';
fprintf('           -------------Q1_PCA-----------\n');
fprintf('           Loading dataset...\n');

training_matrix = [];
for training_index = 1:length(training_struct)
	training_matrix(:,training_index) = training_struct(training_index).training_feature_vector(:);
end

training_mean = mean(training_matrix')';
training_matrix = training_matrix - repmat(training_mean,1,size(training_matrix,2));
[training_matrix_U,training_matrix_S,~] = svd(training_matrix);

PCs = 10;
set(figure(1),'name','Q1-(a). First 10 Eigenfaces');
set(gcf,'Position',get(0,'Screensize')); 

for face_index = 1:PCs 
    eigen_face_matrix = reshape(training_matrix_U(:,face_index),dimensions(1),dimensions(2));
    subplot(2,PCs/2,face_index), imshow((eigen_face_matrix - min(min(eigen_face_matrix)))/(max(max(eigen_face_matrix))- min(min(eigen_face_matrix)))), title(sprintf('Eigenface No.%d (PCs = %d)',face_index,face_index));
end
fprintf('           Ploting first 10 eigenfaces...\n');

dimension_options=[1 2 3 5 10 20 30 40 50 75 100 125 150 175 200];
pca_accuracy=zeros(1,length(dimension_options));

for testing_index = 1:length(testing_struct)
	testing_matrix(:,testing_index) = testing_struct(testing_index).testing_feature_vector(:);
end

testing_mean = mean(testing_matrix')';
testing_matrix = testing_matrix - repmat(testing_mean,1,size(testing_matrix,2));
training_trans_matrix = (training_matrix_U') * training_matrix;
testing_trans_matrix = (training_matrix_U') * testing_matrix;

fprintf('           Computing accuracy as follows:\n');
accuracy_dim=0;
for i_PCA = 1:length(dimension_options)
    accuracy_dim = accuracy_dim+1;
    dimension_index = dimension_options(1,i_PCA);
    test_num = 0;
    agree_num = 0;
    for test_index = 1:length(testing_struct)
            test_num =  test_num + 1;
            mini_distance = realmax;
            mini_dis_index = -1;
            for train_index = 1:length(training_struct)
                now_distance = sqrt(sum((testing_trans_matrix(1:dimension_index,test_index)-training_trans_matrix(1:dimension_index,train_index)).^2));
                if now_distance < mini_distance
                   mini_distance = now_distance;
                   mini_dis_index = train_index;
                end
            end
            if testing_struct(test_index).testing_class_ID == training_struct(mini_dis_index).training_class_ID
               agree_num = agree_num + 1;
            end
    end
    fprintf(1,'           Accuracy (Dimension=%d) = %.2f%%\n',dimension_options(1,accuracy_dim),100*agree_num/test_num);
    pca_accuracy(1,accuracy_dim) = agree_num/test_num;
end
fprintf('           Ploting accuracy v.s no. of PCs...\n');
set(figure(2),'name','Q1-(b). Accuracy v.s No. of PCs');
plot(dimension_options,100*pca_accuracy),title('PCA: accuracy v.s number of PCs'), xlabel('Number of PCs'), ylabel('Accuracy(%)'), grid on;
fprintf('           --------------DONE!-----------\n\n');