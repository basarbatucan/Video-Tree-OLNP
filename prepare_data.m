close all
clear 
clc

% parameters
video_set_name = 'UCSDped2_dae.mat';
alpha = 0.25;

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
frame_n = data.frame_n';
image_paths = data.ori_im_path';

% save actual data
x = zeros(N, 32);
for i=1:32
    x(:,i) = data.(features{k-1+i})';
end
y = data.label';

% select min event treshold for y
anomaly_index = y>alpha;
y(anomaly_index) = 1;
y(~anomaly_index) = -1;

% shuffle data
shuffle_index = randperm(N);
x = x(shuffle_index, :);
y = y(shuffle_index);
frame_n = frame_n(shuffle_index);
image_paths = image_paths(shuffle_index);
object_coords = object_coords(shuffle_index, :);

% save all files to mat
save(['./data/processed/',video_set_name], 'x', 'y', '-v7.3');
save(['./data/processed/meta_',video_set_name], 'frame_n', 'image_paths', 'object_coords', '-v7.3');