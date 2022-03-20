clear
close all
clc

% Define pipeline variables
video_set_name = 'UCSDped2_dae.mat';
input_data_dir = ['./data/processed/',video_set_name];
input_meta_data_dir = ['./data/processed/meta_',video_set_name];

val_size = 0.15;
test_size = 0.15;
augmentation_size = 150e3;
test_repeat = 100;
cross_val_MC = 8;
max_x = 360; % you should select the correct x size and y size
max_y = 240; % space partitioning is done based on this
tfpr = 0.01;

% Define model hyper-parameter space
hyperparams.eta_init = 0.01;
hyperparams.beta_init = 100;
hyperparams.gamma = 1;
hyperparams.sigmoid_h = -1;
hyperparams.lambda = 0;
hyperparams.tree_depth = 6;
hyperparams.split_prob = 0.5;
hyperparams.node_loss_constant = 1;

% generate hyper-parameter space 
hyperparam_space = utility_functions.generate_hyperparameter_space_Video_Tree_OLNP(hyperparams);
hyperparam_number = length(hyperparam_space);
cross_val_scores = zeros(cross_val_MC, hyperparam_number);

% Read Data
data = load(input_data_dir);
meta_data = load(input_meta_data_dir);
[X_train, X_val, X_test, ...
 frames_train, frames_val, frames_test, ...
 image_paths_train, image_paths_val, image_paths_test, ...
 coords_train, coords_val, coords_test, ...
 y_train, y_val, y_test] = utility_functions.train_val_test_split(data.x, data.y, meta_data.frame_n, meta_data.image_paths, meta_data.object_coords, val_size, test_size);
n_features = size(X_train, 2);

% cross validation
if hyperparam_number>1
    
    % force hyperparameter tuning
    X_train_ = X_train;
    frames_train_ = frames_train;
    image_paths_train_ = image_paths_train;
    coords_train_ = coords_train;
    y_train_ = y_train;
    
    X_val_ = X_val;
    frames_val_ = frames_val;
    image_paths_val_ = image_paths_val;
    coords_val_ = coords_val;
    y_val_ = y_val;
    
    % normalization
    [X_train_, mu_train, sigma_train] = zscore(X_train_);
    for i=1:n_features
        X_val_(:,i) = (X_val_(:,i)-mu_train(i))/sigma_train(i);
    end
    
    % compare cross validations
    for i=1:length(hyperparam_space)
        parfor j=1:cross_val_MC
            
            eta_init = hyperparam_space{i}.eta_init;
            beta_init = hyperparam_space{i}.beta_init;
            gamma = hyperparam_space{i}.gamma;
            sigmoid_h = hyperparam_space{i}.sigmoid_h;
            lambda = hyperparam_space{i}.lambda;
            tree_depth = hyperparam_space{i}.tree_depth;
            split_prob = hyperparam_space{i}.split_prob;
            node_loss_constant = hyperparam_space{i}.node_loss_constant;
            
            % load the model
            model = Video_Tree_OLNP(eta_init, beta_init, gamma, sigmoid_h, lambda, tree_depth, split_prob, node_loss_constant, n_features, tfpr, max_x, max_y);
            
            % augmentation (also includes shuffling)
            [X_train__, y_train__, coords_train__] = utility_functions.augment_data(X_train_, y_train_, coords_train_, augmentation_size);
            
            % train the model
            model = model.train(X_train__, coords_train__, y_train__, X_val_, coords_val_, y_val_, 1);

            % evaluate NP score
            tpr = model.tpr_test_array_(end);
            fpr = model.fpr_test_array_(end);
            NP_score = utility_functions.get_NP_score(tpr, fpr, tfpr);
            cross_val_scores(j,i) = NP_score;
            
        end
    end
    
    % make decision based on mean of the NP scores
    cross_val_scores_ = mean(cross_val_scores);
    
    % find out the best hyperparameter set
    % for NP score, lesser is better
    [~, target_hyperparameter_index] = min(cross_val_scores_);
    
    % select optimum hyperparameters
    eta_init = hyperparam_space{target_hyperparameter_index}.eta_init;
    beta_init = hyperparam_space{target_hyperparameter_index}.beta_init;
    gamma = hyperparam_space{target_hyperparameter_index}.gamma;
    sigmoid_h = hyperparam_space{target_hyperparameter_index}.sigmoid_h;
    lambda = hyperparam_space{target_hyperparameter_index}.lambda;
    tree_depth = hyperparam_space{target_hyperparameter_index}.tree_depth;
    split_prob = hyperparam_space{target_hyperparameter_index}.split_prob;
    node_loss_constant = hyperparam_space{target_hyperparameter_index}.node_loss_constant;
    
else
    
    % there is only one hyperparameter defined
    eta_init = hyperparam_space{1}.eta_init;
    beta_init = hyperparam_space{1}.beta_init;
    gamma = hyperparam_space{1}.gamma;
    sigmoid_h = hyperparam_space{1}.sigmoid_h;
    lambda = hyperparam_space{1}.lambda;
    tree_depth = hyperparam_space{1}.tree_depth;
    split_prob = hyperparam_space{1}.split_prob;
    node_loss_constant = hyperparam_space{1}.node_loss_constant;
    
end

%% training
% since hyperparameter tuning is completed, merge train and val
X_train = [X_train;X_val];
y_train = [y_train;y_val];
coords_train = [coords_train;coords_val];
[X_train, mu_train, sigma_train] = zscore(X_train);
for i=1:n_features
    X_test(:,i) = (X_test(:,i)-mu_train(i))/sigma_train(i);
end

% Preprocessing
[X_train, coords_train, y_train] = utility_functions.augment_data(X_train, coords_train, y_train, augmentation_size);

% load the model
model = Video_Tree_OLNP(eta_init, beta_init, gamma, sigmoid_h, lambda, tree_depth, split_prob, node_loss_constant, n_features, tfpr, max_x, max_y);

% train the model
model = model.train(X_train, coords_train, y_train, X_test, coords_test, y_test, 100);

% plot the results
model.plot_results();