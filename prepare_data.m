close all
clear 
clc

% parameters
video_set_name = 'UCSDped2_dae.mat';
%video_set_name = 'SUrveillance_dae.mat';
%video_set_name = 'UCSDped1_dae.mat';
%video_set_name = 'ShangaiTech_dae.mat';
alpha = 0.65;

% read raw data
data = load(['./data/raw/',video_set_name]);
features = fieldnames(data);
for i=1:length(features)
    if strcmp(features{i},'dae_latent_feature_00')
        k=i;
        break
    end
end

% get number of samples
N = length(data.xmin);

% save meta data
object_coords = zeros(N, 5);
object_coords(:,1) = data.xmin';
object_coords(:,2) = data.ymin';
object_coords(:,3) = data.xmax';
object_coords(:,4) = data.ymax';
train_valid_test = data.train_valid_test';
frame_n = data.frame_n';
image_paths = data.ori_im_path';

% save actual data
x = zeros(N, 33);
% 32 latent variables + 1 spnr
for i=1:33
    x(:,i) = data.(features{k-1+i})';
end
y = data.label';

% select min event treshold for y
anomaly_index = y>alpha;
y(anomaly_index) = 1;
y(~anomaly_index) = -1;

% save all files to mat
save(['./data/processed/',video_set_name], 'x', 'y', '-v7.3');
save(['./data/processed/meta_',video_set_name], 'frame_n', 'image_paths', 'object_coords', 'train_valid_test','-v7.3');