function model = single_experiment(tfpr, data_name, test_repeat, optimized_params)

    % Define pipeline variables
    video_set_name = sprintf('%s_dae.mat', data_name);
    input_data_dir = ['./data/processed/',video_set_name];
    input_meta_data_dir = ['./data/processed/meta_',video_set_name];
    
    augmentation_size = 150e3;
    cross_val_MC = 8;
    %x and y scales
    % read 1 image from test data to determine the size of the feature
    % space
    tmp_name = dir(['./data/raw/', data_name, '_example.*']);
    tmp_im = imread(fullfile(tmp_name.folder, tmp_name.name));
    [max_y, max_x, ~] = size(tmp_im);

    % Define model hyper-parameter space
    hyperparams.eta_init = 0.05;
    hyperparams.beta_init = [5e2];
    hyperparams.gamma = 1;
    hyperparams.sigmoid_h = -2;
    hyperparams.lambda = 0;
    hyperparams.tree_depth = [8];
    hyperparams.split_prob = 0.5;
    hyperparams.node_loss_constant = [2];
    %hyperparams.node_loss_constant = [1e-1]; % decrease constants by 10 if tree convergence fails

    % generate hyper-parameter space 
    hyperparam_space = utility_functions.generate_hyperparameter_space_Video_Tree_OLNP(hyperparams);
    hyperparam_number = length(hyperparam_space);
    cross_val_scores = zeros(cross_val_MC, hyperparam_number);

    % Read Data
    data = load(input_data_dir);
    meta_data = load(input_meta_data_dir);
    
    % update train test validation for NP formulation
    new_train_valid_test = utility_functions.update_train_valid_test(meta_data.train_valid_test, data.y);
    
    [X_train, X_val, X_test, ...
     frames_train, frames_val, frames_test, ...
     image_paths_train, image_paths_val, image_paths_test, ...
     coords_train, coords_val, coords_test, ...
     y_train, y_val, y_test] = utility_functions.train_val_test_split(data.x, data.y, new_train_valid_test, meta_data.frame_n, meta_data.image_paths, meta_data.object_coords);
    n_features = size(X_train, 2);
    
    close all;
    
    % cross validation
    if isempty(optimized_params)
        
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
            fprintf('Hyperparameter space size: %d\n', hyperparam_number);
            for i=1:length(hyperparam_space)
                
                tuning_tstart = tic;
                
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
                    [X_train__, coords_train__, y_train__] = utility_functions.augment_data(X_train_, coords_train_, y_train_, augmentation_size);

                    % train the model
                    model = model.train(X_train__, coords_train__, y_train__, X_val_, coords_val_, y_val_, 1);

                    % evaluate NP score
                    tpr = model.tpr_test_array_(end);
                    fpr = model.fpr_test_array_(end);
                    NP_score = utility_functions.get_NP_score(tpr, fpr, tfpr);
                    cross_val_scores(j,i) = NP_score;

                end
                
                tuning_tend = toc(tuning_tstart);
                fprintf('Time elapsed for testing hyperparameter set %d: %.3f\n', i, tuning_tend);
                
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
        
        % load the model (this case is only used to save the parameters)
        model = Video_Tree_OLNP(eta_init, beta_init, gamma, sigmoid_h, lambda, tree_depth, split_prob, node_loss_constant, n_features, tfpr, max_x, max_y);
        
    else
        
        % model is already optimized
        eta_init = optimized_params.eta_init;
        beta_init = optimized_params.beta_init;
        gamma = optimized_params.gamma;
        sigmoid_h = optimized_params.sigmoid_h;
        lambda = optimized_params.lambda;
        tree_depth = optimized_params.tree_depth;
        split_prob = optimized_params.split_prob;
        node_loss_constant = optimized_params.node_loss_constant;
        
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
        model = model.train(X_train, coords_train, y_train, X_test, coords_test, y_test, test_repeat);

        % plot the results
        model.plot_results();
    
    end

end