classdef utility_functions
    
    methods (Static)
        
        % common functions
        function [X_train, X_val, X_test, ...
                  frames_train, frames_val, frames_test, ...
                  image_paths_train, image_paths_val, image_paths_test, ...
                  coords_train, coords_val, coords_test, ...
                  y_train, y_val, y_test] = train_val_test_split(X, y, frames, image_paths, coords, val_size, test_size)
            
            if (nargin<3) || (val_size>=1) || (test_size>=1)
                val_size = 0.15;
                test_size = 0.15;
            end
            
            N = length(y);
            shuffle_index = randperm(N);
            X = X(shuffle_index, :);                                       % shuffle the data
            y = y(shuffle_index, :);                                       % shuffle the data
            frames = frames(shuffle_index, :);                             % shuffle the data
            image_paths = image_paths(shuffle_index, :);                   % shuffle the data
            coords = coords(shuffle_index, :);                             % shuffle the data
            
            val_N = round(N*val_size);
            test_N = round(N*test_size);
            train_index = 1:N-(val_N+test_N);
            val_index = N-(val_N+test_N)+1:N-test_N;
            test_index = N-test_N+1:N;
            
            X_train = X(train_index, :);
            frames_train = frames(train_index, :);
            image_paths_train = image_paths(train_index, :);
            coords_train = coords(train_index, :);
            y_train = y(train_index, :);
            
            X_val = X(val_index, :);
            frames_val = frames(val_index, :);
            image_paths_val = image_paths(val_index, :);
            coords_val = coords(val_index, :);
            y_val = y(val_index, :);
            
            X_test = X(test_index, :);
            frames_test = frames(test_index, :);
            image_paths_test = image_paths(test_index, :);
            coords_test = coords(test_index, :);
            y_test = y(test_index, :);
            
        end
        
        % augmenting the input data for convergence of the online model
        function [augmented_x, augmented_coords, augmented_y] = augment_data(x, coords, y, augmentation_size)

            if (nargin<4)
                augmentation_size = 150e3;
            end
            
            [N,M] = size(x);
            if N<augmentation_size
                % concat necessary
                concat_time = ceil(augmentation_size/N);
                augmented_x = zeros(concat_time*N, M);
                augmented_coords = zeros(concat_time*N, 5);
                augmented_y = zeros(concat_time*N, 1);
                for i=1:concat_time
                    start_i = (i-1)*N+1;
                    end_i = i*N;
                    shuffle_i = randperm(length(y));
                    augmented_x(start_i:end_i, :) = x(shuffle_i, :);
                    augmented_coords(start_i:end_i, :) = coords(shuffle_i, :);
                    augmented_y(start_i:end_i, :) = y(shuffle_i, :);
                end
            else
                % no concat
                augmented_x = x;
                augmented_coords = coords;
                augmented_y = y;
            end

        end
        
        % derivative of sigmoid function
        function ret = deriv_sigmoid_loss(z, h)
            sigmoid_loss_x = utility_functions.sigmoid_loss(z, h);
            ret = h*(1-sigmoid_loss_x)*sigmoid_loss_x;
        end

        % sigmoid function
        function ret = sigmoid_loss(z,h)
            ret = 1/(1+exp(-h*z));
        end
        
        % calculate NP score
        function ret = get_NP_score(tpr, fpr, tfpr)
            ret = max(fpr, tfpr)/tfpr-tpr;
        end
        
        % get all parameters and generate hyperparameter space
        % gridsearch cross validation
        function hyperparameter_space = generate_hyperparameter_space_Video_Tree_OLNP(parameters)
            
            eta_init_space = parameters.eta_init;
            beta_init_space = parameters.beta_init;
            gamma_space = parameters.gamma;
            sigmoid_h_space = parameters.sigmoid_h;
            lambda_space = parameters.lambda;
            tree_depth_space = parameters.tree_depth;
            split_prob_space = parameters.split_prob;
            node_loss_constant_space = parameters.node_loss_constant;
            n1 = length(eta_init_space);
            n2 = length(beta_init_space);
            n3 = length(gamma_space);
            n4 = length(sigmoid_h_space);
            n5 = length(lambda_space);
            n6 = length(tree_depth_space);
            n7 = length(split_prob_space);
            n8 = length(node_loss_constant_space);
            
            % create hyperparameter space
            N = n1*n2*n3*n4*n5*n6*n7*n8;
            hyperparameter_space = cell(N, 1);
            
            % fill the hyperparameter space
            hyper_param_i=1;
            for i_01=1:n1
                for i_02=1:n2
                    for i_03=1:n3
                        for i_04=1:n4
                            for i_05=1:n5
                                for i_06=1:n6
                                    for i_07=1:n7
                                        for i_08=1:n8
                                            hyperparameter_space{hyper_param_i}.eta_init = eta_init_space(i_01);
                                            hyperparameter_space{hyper_param_i}.beta_init = beta_init_space(i_02);
                                            hyperparameter_space{hyper_param_i}.gamma = gamma_space(i_03);
                                            hyperparameter_space{hyper_param_i}.sigmoid_h = sigmoid_h_space(i_04);
                                            hyperparameter_space{hyper_param_i}.lambda = lambda_space(i_05);
                                            hyperparameter_space{hyper_param_i}.tree_depth = tree_depth_space(i_06);
                                            hyperparameter_space{hyper_param_i}.split_prob = split_prob_space(i_07);
                                            hyperparameter_space{hyper_param_i}.node_loss_constant = node_loss_constant_space(i_08);
                                            hyper_param_i = hyper_param_i+1;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
            
        end
        
    end
    
end