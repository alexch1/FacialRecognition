clear;
load 'preprocess.mat';
fprintf('           -------------Q2_NMF-----------\n');
fprintf('           Loading dataset...\n');

dim_max = 50;
V = zeros(length(wholedata_struct(1).feature_vector(:)),length(wholedata_struct));
for data_index = 1:length(wholedata_struct)
	V(:,data_index) = wholedata_struct(data_index).feature_vector(:);
end

W = rand(length(wholedata_struct(1).feature_vector(:)),dim_max);
H = rand(dim_max,length(wholedata_struct));
NMF_iteration = 50;
H_1 = zeros(size(H));
W_1 = zeros(size(W));
H_2 = H;
W_2 = W;    
for i = 1:NMF_iteration

    if mod(i,2) == 1
        H_update_num = W_2' * V;
        H_update_den = (W_2' * W_2) * H_2;
        H_1 = H_2 .* H_update_num ./ H_update_den;
        W_update_num = V * H_2';
        W_update_den = W_2 * (H_2 * H_2');
        W_1 = W_2 .* W_update_num ./ W_update_den;

    else

        H_update_num = W_1' * V;
        H_update_den = (W_1' * W_1) * H_2;
        H_2 = H_1 .* H_update_num ./ H_update_den;
        W_update_num = V * H_1';
        W_update_den = W_1 * (H_1 * H_1');
        W_2 = W_1 .* W_update_num ./ W_update_den;
    end
    
end
W=W_2;
H=H_2;

fprintf('           Ploting NFM bases...\n');
bases_num = 50;
set(figure(3),'name','Q2-(a). NMF bases');
set(gcf,'Position',get(0,'Screensize'));
for nmf_index = 1:bases_num 
    nmf_matrix = reshape(W(:,nmf_index),dimensions(1),dimensions(2));
    subplot(10,bases_num/10,nmf_index), imshow((nmf_matrix - min(min(nmf_matrix)))/(max(max(nmf_matrix)) - min(min(nmf_matrix))));
end


%run twice again
for random_index=4:5
W = rand(length(wholedata_struct(1).feature_vector(:)),dim_max);
H = rand(dim_max,length(wholedata_struct));
NMF_iteration = 50;
H_1 = zeros(size(H));
W_1 = zeros(size(W));
H_2 = H;
W_2 = W;    
for i = 1:NMF_iteration

    if mod(i,2) == 1
        H_update_num = W_2' * V;
        H_update_den = (W_2' * W_2) * H_2;
        H_1 = H_2 .* H_update_num ./ H_update_den;
        W_update_num = V * H_2';
        W_update_den = W_2 * (H_2 * H_2');
        W_1 = W_2 .* W_update_num ./ W_update_den;

    else

        H_update_num = W_1' * V;
        H_update_den = (W_1' * W_1) * H_2;
        H_2 = H_1 .* H_update_num ./ H_update_den;
        W_update_num = V * H_1';
        W_update_den = W_1 * (H_1 * H_1');
        W_2 = W_1 .* W_update_num ./ W_update_den;
    end
    
end

W=W_2;
H=H_2;

fprintf('           Ploting NFM bases_random...\n');
bases_num = 50;
set(figure(random_index),'name',strcat('Q2-(b). NMF bases_random #',int2str(random_index-3)));
set(gcf,'Position',get(0,'Screensize'));
for nmf_index = 1:bases_num 
    nmf_matrix = reshape(W(:,nmf_index),dimensions(1),dimensions(2));
    subplot(10,bases_num/10,nmf_index), imshow((nmf_matrix - min(min(nmf_matrix)))/(max(max(nmf_matrix))- min(min(nmf_matrix))));
end
end

fprintf('           --------------DONE!-----------\n\n');