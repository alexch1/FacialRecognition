clear;
load 'preprocess.mat';
fprintf('           -------------Q4_GMM-----------\n');
fprintf('           Loading dataset...\n');

gmm_matrix = [];
for data_index = 1:length(wholedata_struct)
	gmm_matrix(:,data_index) = wholedata_struct(data_index).feature_vector(:);
end

gmm_data_mean = mean(gmm_matrix,2);
gmm_matrix = gmm_matrix - repmat(gmm_data_mean,1,size(gmm_matrix,2));
[gmm_matrix_u,~,~] = svd(gmm_matrix);

gmm_pca_bases = 50;
gmm_matrix_reduced = (gmm_matrix_u(:,1:gmm_pca_bases)') * gmm_matrix;

gmm_components = 10;
gmm_prior = ones(gmm_components,1)/gmm_components;
gmm_mean = gmm_matrix_reduced(:,randperm(size(gmm_matrix_reduced,2),gmm_components));
gmm_mem = zeros(size(gmm_matrix_reduced,2),gmm_components);

for i = 1:gmm_components
    gmm_cov(:,:,i) = 50*eye(gmm_pca_bases);
end

old_gmm_mean = gmm_mean;

for gmm_loop=1:2
    
    for component_index = 1:gmm_components
        gmm_mem(:,component_index) = gmm_prior(component_index) * mvnpdf(gmm_matrix_reduced', gmm_mean(:,component_index)', gmm_cov(:,:,component_index));
    end
    
    gmm_mem = gmm_mem ./ repmat(sum(gmm_mem,2),1,gmm_components);
    gmm_prior = mean(gmm_mem)';
    gmm_mean = (gmm_matrix_reduced * gmm_mem) ./ repmat(sum(gmm_mem),gmm_pca_bases,1);
    gmm_cov = zeros(gmm_pca_bases,gmm_pca_bases,gmm_components);
    
    for component_index = 1:gmm_components
        for data_index = 1:size(gmm_matrix_reduced,2)
            gmm_cov(:,:,component_index) = gmm_cov(:,:,component_index) + gmm_mem(data_index,component_index) * ((gmm_matrix_reduced(:,data_index) - gmm_mean(:,component_index)) * (gmm_matrix_reduced(:,data_index) - gmm_mean(:,component_index))');
        end
        gmm_cov(:,:,component_index) = gmm_cov(:,:,component_index) / sum(gmm_mem(:,component_index));
    end
    old_gmm_mean = gmm_mean;
    
end
gmm_reformed_u = gmm_matrix_u(:,1:gmm_pca_bases) * gmm_mean;

fprintf('           Ploting 10 component centres of GMM...\n');
set(figure(8),'name','Q4-(a). Component centres');
set(gcf,'Position',get(0,'Screensize')); 
for gmm_face_index = 1:gmm_components
    gmm_face_matrix = zeros(dimensions);
    gmm_face_matrix = reshape(gmm_reformed_u(:,gmm_face_index),dimensions(1),dimensions(2));
    subplot(2,gmm_components/2,gmm_face_index), imshow((gmm_face_matrix - min(min(gmm_face_matrix)))/(max(max(gmm_face_matrix)) - min(min(gmm_face_matrix)))), title(sprintf('GMM face No.%d',gmm_face_index));
end

fprintf('           --------------DONE!-----------\n\n');