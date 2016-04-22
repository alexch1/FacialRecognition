clear;
load 'preprocess.mat';
fprintf('           -------------Q3_LDA-----------\n');
fprintf('           Loading dataset...\n');

lda_train_matrix = [];
lda_train_class = [];
lda_train_index = 1;
lda_pca_bases = 1000;

for data_index = 1:length(wholedata_struct)
    if wholedata_struct(data_index).type
        lda_train_matrix(:,lda_train_index) = wholedata_struct(data_index).feature_vector(:);
        lda_train_class(lda_train_index) = wholedata_struct(data_index).class_ID;
        lda_train_index = lda_train_index + 1;
    end
end

lda_train_mean = mean(lda_train_matrix,2);
lda_train_matrix = lda_train_matrix - repmat(lda_train_mean,1,size(lda_train_matrix,2));
[lda_train_matrix_u,~,~] = svd(lda_train_matrix);

lda_train_matrix_reduced = (lda_train_matrix_u(:,1:lda_pca_bases)') * lda_train_matrix;

lda_w_class_scatter = zeros(lda_pca_bases,lda_pca_bases);
lda_b_class_scatter = zeros(lda_pca_bases,lda_pca_bases);
lda_tol_scatter = zeros(lda_pca_bases,lda_pca_bases);
lda_mix_mean = mean(lda_train_matrix_reduced,2);

for train_index = 1:size(lda_train_matrix_reduced,2)
    lda_tol_scatter = lda_tol_scatter + (lda_train_matrix_reduced(:,train_index) - lda_mix_mean) * (lda_train_matrix_reduced(:,train_index) - lda_mix_mean)';
end

lda_tol_scatter = lda_tol_scatter / size(lda_train_matrix_reduced,2);
num_class_members = 0;
now_class_id = -1;
class_mean_vector = zeros(size(lda_train_matrix_reduced,1),1);
lda_train_class(end + 1) = -1;

for train_index = 1:size(lda_train_matrix_reduced,2) + 1
    if now_class_id ~= lda_train_class(train_index)
        class_mean_vector = class_mean_vector / num_class_members;
        if num_class_members > 0
          lda_b_class_scatter = lda_b_class_scatter + (num_class_members / size(lda_train_matrix_reduced,2)) * (class_mean_vector - lda_mix_mean) * (class_mean_vector - lda_mix_mean)';
        end
        num_class_members = 0;
        class_mean_vector = zeros(size(lda_train_matrix_reduced,1),1);
        now_class_id = lda_train_class(train_index);
    end
    
    if train_index > size(lda_train_matrix_reduced,2)
        break;
    end
    class_mean_vector = class_mean_vector + lda_train_matrix_reduced(:,train_index);
    num_class_members = num_class_members + 1;
    
end

lda_w_class_scatter = lda_tol_scatter - lda_b_class_scatter;
[lda_train_matrix_reduced_u,lda_train_matrix_reduced_s,~] = svd(lda_w_class_scatter \ lda_b_class_scatter);

lda_train_matrix_reformed_u = lda_train_matrix_u(:,1:lda_pca_bases) * lda_train_matrix_reduced_u;

fprintf('           Ploting first 10 fisherfaces...\n');
fisher_faces = 10;
set(figure(6),'name','Q3-(a). First 10 Fisherfaces');
set(gcf,'Position',get(0,'Screensize')); 
for fisher_face_index = 1:fisher_faces
    fisher_faces_matrix = zeros(dimensions);
    fisher_faces_matrix = reshape(lda_train_matrix_reformed_u(:,fisher_face_index),dimensions(1),dimensions(2));
    subplot(2,fisher_faces/2,fisher_face_index), imshow((fisher_faces_matrix - min(min(fisher_faces_matrix)))/(max(max(fisher_faces_matrix)) - min(min(fisher_faces_matrix)))), title(sprintf('Fisherface No.%d',fisher_face_index));
end

lda_dimension_options=[1 2 3 5 10 20 30 40 50 75 100 125 150 175 200 225 250 275 300 325 350 379];
lda_accuracy=zeros(1,length(lda_dimension_options));
data_matrix = [];
for data_index = 1:length(wholedata_struct)
	data_matrix(:,data_index) = wholedata_struct(data_index).feature_vector(:);
end

data_mean = mean(data_matrix,2);
data_matrix = data_matrix - repmat(data_mean,1,size(data_matrix,2));
data_transform_matrix = lda_train_matrix_reduced_u' * lda_train_matrix_u(:,1:lda_pca_bases)' * data_matrix;

fprintf('           Computing accuracy as follows:\n');
accuracy_dim=0;
for i_lda = 1:length(lda_dimension_options)
    accuracy_dim = accuracy_dim+1;
    dimension_index = lda_dimension_options(1,i_lda);
    test_num = 0;
    agree_num = 0;
    for test_index = 1:length(wholedata_struct)
        if wholedata_struct(test_index).type == false
            test_num =  test_num + 1;
            mini_distance = realmax;
            mini_dis_index = -1;
            for train_index = 1:length(wholedata_struct)
                if wholedata_struct(train_index).type == true
                    now_distance = sqrt(sum((data_transform_matrix(1:dimension_index,test_index)-data_transform_matrix(1:dimension_index,train_index)).^2));
                    if now_distance < mini_distance
                        mini_distance = now_distance;
                        mini_dis_index = train_index;
                    end 
                end
            end
            if wholedata_struct(test_index).class_ID == wholedata_struct(mini_dis_index).class_ID
                agree_num = agree_num + 1;
            end  
        end
    end
    fprintf(1,'           Accuracy (Dimension=%d) = %.2f%%\n',lda_dimension_options(1,accuracy_dim),100*agree_num/test_num);
    lda_accuracy(1,accuracy_dim) = agree_num/test_num;
    
end

fprintf('           Ploting accuracy v.s no. of fisher components...\n');
set(figure(7),'name','Q3-(b). LDA: accuracy vs. No. of fisher components');
plot(lda_dimension_options,100*lda_accuracy), title('Accuracy vs. Number of Fisher components'), xlabel('Number of Fisher components'), ylabel('Accuracy(%)'), grid on;

fprintf('           --------------DONE!-----------\n\n');