
"""
Training utilities and cross-validation framework.
"""

def initialize_results():
    return {
        "fold_model": [], "train_acc": [], "val_acc": [], "accuracy_scores": [], "recall_scores": [],"precision_scores": [], "f1_scores": [], "roc_auc_scores": [], "specificity_scores": [],
        "mae": [], "mse": [],"fold_train_loss": [],"fold_train_auc": [],"fold_train_acc": [],"fold_val_loss": [],"fold_val_auc": [],"fold_val_acc": [],
        "fold_val_recall": [], "fold_y_true": [],"fold_y_pred": [],"fold_y_pred_proba": [],"calibration_data": [], "feature_importance": [], "conf_matrix": [], 
        "class_report": [],"train_loss": [],"train_acc": [],"val_loss": [],"val_acc": [],"roc_data": []
    }

def prepare_tf_dataset(X_matrix, X_features1, X_features2, y, batch_size=32, shuffle=True):
    # Calculate number of complete batches (no partial batches)
    n_samples = len(y)
    n_complete_batches = n_samples // batch_size
    
    # Only use complete batches
    indices = np.arange(n_complete_batches * batch_size)
    
    # Create dataset from truncated data
    dataset = tf.data.Dataset.from_tensor_slices((
        (X_matrix[indices], X_features1[indices], X_features2[indices]), 
        y[indices]
    ))
    dataset = dataset.map(
        lambda x, y: ((tf.cast(x[0], tf.float32), tf.cast(x[1], tf.float32), tf.cast(x[2], tf.float32)), tf.cast(y, tf.float32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_complete_batches * batch_size)
    
    # Use fixed-size batches with no remainder
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset
    
def prepare_tf_dataset(X_matrix, X_features1, X_features2, y, batch_size=16, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(((X_matrix, X_features1, X_features2), y))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(y), reshuffle_each_iteration=True)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
    
def get_train_val_test_indices(participant_ids, train_val_idx, test_idx, y_labels):
    """
    Split indices into train and validation sets while maintaining class distribution
    and preventing participant-level data leakage.
    """
    # Get unique participants in the train_val set
    train_val_participants = np.unique(participant_ids[train_val_idx])
    test_participants = np.unique(participant_ids[test_idx])
    
    print(f"Total participants: Train+Val: {len(train_val_participants)}, Test: {len(test_participants)}")
    
    # Create participant-level labels for stratification
    participant_labels = []
    for participant in train_val_participants:
        idx = np.where(participant_ids == participant)[0]
        # Label a participant as positive if any of their samples are positive
        label_values = y_labels[idx]
        participant_labels.append(int(np.any(label_values == 1)))
    
    participant_labels = np.array(participant_labels)
    
    # Print class distribution at participant level
    pos_ratio = np.mean(participant_labels)
    print(f"Class distribution at participant level: Positive: {pos_ratio:.4f}, Negative: {1-pos_ratio:.4f}")
    
    # Split participants using stratified split
    train_participants, val_participants = train_test_split(
        train_val_participants,
        test_size=1/7,  # Keep the same ratio as before
        random_state=42,
        stratify=participant_labels
    )
    
    print(f"{len(train_participants)}, {len(val_participants)}, {len(test_participants)}, participants in Train, Validation, and Test sets")
    
    # Convert participant lists back to sample indices
    train_idx = np.where(np.isin(participant_ids, train_participants))[0]
    val_idx = np.where(np.isin(participant_ids, val_participants))[0]
    
    # Verify class distribution
    train_pos_ratio = np.mean(y_labels[train_idx] == 1)
    val_pos_ratio = np.mean(y_labels[val_idx] == 1)
    test_pos_ratio = np.mean(y_labels[test_idx] == 1)
    
    print(f"Sample-level class distribution - Train: {train_pos_ratio:.4f}, Val: {val_pos_ratio:.4f}, Test: {test_pos_ratio:.4f}")
    
    return train_idx, val_idx
    
def validate_dataset(dataset):
    """Verify all batches have consistent non-zero shapes"""
    for batch_idx, (inputs, _) in enumerate(dataset):
        for tensor_idx, tensor in enumerate(inputs):
            shape = tf.shape(tensor)
            if tf.reduce_any(tf.equal(shape, 0)):
                raise ValueError(f"Empty dimension detected in batch {batch_idx}, tensor {tensor_idx}: {shape}")
    return True
    
def train_and_evaluate_model(fold, train_idx, val_idx, X_matrix, X_daily_features, X_individual_features, y_labels, num_epochs=40, batch_s=16, num_patience=20, monitored='val_loss'):
    # Apply median imputation to features
    X_daily_train = X_daily_features[train_idx]
    X_daily_val = X_daily_features[val_idx]
    X_individual_train = X_individual_features[train_idx]
    X_individual_val = X_individual_features[val_idx]
    
    # Perform median imputation
    X_daily_train_imputed, X_daily_val_imputed, _ = median_impute_features(X_daily_train, X_daily_val)
    X_individual_train_imputed, X_individual_val_imputed, _ = median_impute_features(X_individual_train, X_individual_val)
    
    # Create the improved model
    model = create_model(X_matrix.shape[1], X_daily_features.shape[1], X_individual_features.shape[1])

    # Setup callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitored, patience=num_patience, restore_best_weights=True)
    learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitored, factor=0.5, patience=5, min_lr=1e-7)
    
    # Prepare TF datasets
    train_dataset = prepare_tf_dataset(X_matrix[train_idx], X_daily_train_imputed, X_individual_train_imputed, 
                                      y_labels[train_idx], batch_size=batch_s)
    val_dataset = prepare_tf_dataset(X_matrix[val_idx], X_daily_val_imputed, X_individual_val_imputed, 
                                    y_labels[val_idx], batch_size=batch_s, shuffle=False)
    validate_dataset(train_dataset)
    validate_dataset(val_dataset)
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_labels[train_idx]), y=y_labels[train_idx].ravel())
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]*2}  # Multiply by 2 to emphasize positive class
    print("Class weights: ", class_weight_dict)
    
    # Train the model
    history = model.fit(train_dataset, epochs=num_epochs, verbose=1, validation_data=val_dataset, 
                      callbacks=[early_stopping, learning_rate_scheduler], class_weight=class_weight_dict) #, class_weight=class_weight_dict

    # Clean up
    del train_dataset, val_dataset
    gc.collect()
    
    return model, history
    
def evaluate_performance(model, test_idx, X_matrices, X_daily_features, X_individual_features, y_labels, participant_ids, threshold=0.3):
    # Apply median imputation to test features
    _, _, X_daily_test_imputed = median_impute_features(X_daily_features, test_features=X_daily_features[test_idx])
    _, _, X_individual_test_imputed = median_impute_features(X_individual_features, test_features=X_individual_features[test_idx])
    
    # Get predictions
    y_true_test = y_labels[test_idx].flatten()
    y_pred_test = model.predict([X_matrices[test_idx], X_daily_test_imputed, X_individual_test_imputed])
    
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    for threshold in thresholds:
        y_pred_binary = (y_pred_test >= threshold).astype(int)
        
        if len(np.unique(y_pred_binary)) > 1:
            sens = recall_score(y_true_test, y_pred_binary)
            spec = recall_score(y_true_test, y_pred_binary, pos_label=0)
        else:
            sens = 0 if threshold > 0.5 else 1
            spec = 1 if threshold > 0.5 else 0
        
        sensitivities.append(sens)
        specificities.append(spec)
    
    # Convert to NumPy arrays for vectorized computation
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    # Compute absolute difference
    diff = np.abs(sensitivities - specificities)
    # Find indices where the difference is minimal
    min_diff = np.min(diff)
    candidates = np.where(diff == min_diff)[0]
    # Among those, find the one with highest average sensitivity + specificity
    best_idx = candidates[np.argmax((sensitivities[candidates] + specificities[candidates]) / 2)]
    best_threshold = thresholds[best_idx]
    
    y_pred_test_binary = (y_pred_test >= best_threshold).astype(int)
    
    # 2. ROC Curve Data with Confidence Intervals
    fpr, tpr, _ = roc_curve(y_true_test, y_pred_test)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap confidence intervals for AUC
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    aucs = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true_test), len(y_true_test))
        if len(np.unique(y_true_test[indices])) < 2:
            continue
        auc_bootstrap = roc_auc_score(y_true_test[indices], y_pred_test[indices])
        aucs.append(auc_bootstrap)
    
    aucs = np.array(aucs)
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    
    # 3. Calibration Curve Data
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true_test, y_pred_test, n_bins=10, strategy='quantile' # 'quantile'
    )
    # # Feature Importance Analysis (for tabular features only)
    # # Calculate permutation importance for tabular features
    # def model_predict_wrapper(X_combined):
    #     # Split combined features back into original format
    #     n_daily = X_daily_features.shape[1]
    #     X_daily_subset = X_combined[:, :n_daily]
    #     X_individual_subset = X_combined[:, n_daily:]
        
    #     # Use mean matrices for this subset (since we're only varying tabular features)
    #     mean_matrices = np.mean(X_matrices[test_idx], axis=0, keepdims=True)
    #     mean_matrices = np.repeat(mean_matrices, X_daily_subset.shape[0], axis=0)
        
    #     return model.predict([mean_matrices, X_daily_subset, X_individual_subset]).flatten()
    
    # # Combine features for permutation importance
    # X_combined_test = np.concatenate([X_daily_test_imputed, X_individual_test_imputed], axis=1)
    
    # # Calculate permutation importance
    # perm_importance = permutation_importance(
    #     model_predict_wrapper, X_combined_test, y_true_test, 
    #     n_repeats=10, random_state=42, scoring='roc_auc'
    # )
    
    # # Create feature names
    # daily_feature_names = [f'Daily_Feature_{i}' for i in range(X_daily_features.shape[1])]
    # individual_feature_names = [f'Individual_Feature_{i}' for i in range(X_individual_features.shape[1])]
    # all_feature_names = daily_feature_names + individual_feature_names
    
    # # Per-participant recall analysis
    # df_results = pd.DataFrame({
    #     "participant_id": participant_ids[test_idx].astype(int),
    #     "y_true": y_true_test,
    #     "y_pred_binary": y_pred_test_binary,
    #     "y_pred_proba": y_pred_test
    # })
    # recall_per_participant = df_results.groupby("participant_id").apply(lambda g: recall_score(g["y_true"], g["y_pred_binary"]) if sum(g["y_true"]) > 0 else np.nan)

    return {
        "accuracy_scores": accuracy_score(y_labels[test_idx], y_pred_test_binary),
        "recall_scores": recall_score(y_labels[test_idx], y_pred_test_binary),
        "precision_scores": precision_score(y_labels[test_idx], y_pred_test_binary),
        "f1_scores": f1_score(y_labels[test_idx], y_pred_test_binary),
        "specificity_scores": recall_score(y_labels[test_idx], y_pred_test_binary, pos_label=0),
        "roc_auc_scores": roc_auc_score(y_labels[test_idx], y_pred_test),
        "fold_y_true": y_true_test,
        "fold_y_pred": y_pred_test_binary,
        "fold_y_pred_proba": y_pred_test,
        "roc_data": {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "ci_lower": ci_lower, "ci_upper": ci_upper},
        "calibration_data": {"fraction_of_positives": fraction_of_positives, "mean_predicted_value": mean_predicted_value},
        # "feature_importance": {"feature_names": all_feature_names, "importance_mean": perm_importance.importances_mean,"importance_std": perm_importance.importances_std},
        # "recall_per_participant": recall_per_participant,
        "conf_matrix": confusion_matrix(y_labels[test_idx], y_pred_test_binary)}

def get_target_results(models, num_splits, X_matrices, X_daily_features, X_individual_features, participant_ids, y_labels):

    # Initialize arrays to store predictions from each fold
    y_pred_target_agg = np.zeros((len(y_labels),)) 
    all_predictions = []
    all_feature_names_all = []
    perm_importance_all = []
    # Aggregate predictions from each fold model
    for model in models:
        y_pred_target_fold = model.predict([X_matrices, X_daily_features, X_individual_features])
        y_pred_target_agg += y_pred_target_fold.flatten()  # Accumulate predictions
        all_predictions.append(y_pred_target_fold.flatten())
        # # Feature Importance Analysis (for tabular features only)
        # # Calculate permutation importance for tabular features
        # def model_predict_wrapper(X_combined):
        #     # Split combined features back into original format
        #     n_daily = X_daily_features.shape[1]
        #     X_daily_subset = X_combined[:, :n_daily]
        #     X_individual_subset = X_combined[:, n_daily:]
            
        #     # Use mean matrices for this subset (since we're only varying tabular features)
        #     mean_matrices = np.mean(X_matrices, axis=0, keepdims=True)
        #     mean_matrices = np.repeat(mean_matrices, X_daily_subset.shape[0], axis=0)
            
        #     return model.predict([mean_matrices, X_daily_subset, X_individual_subset]).flatten()
        
        # # Combine features for permutation importance
        # X_combined_test = np.concatenate([X_daily_features, X_individual_features], axis=1)

        # # Calculate permutation importance
        # perm_importance = permutation_importance(
        #     model_predict_wrapper, X_combined_test, y_true_test, 
        #     n_repeats=5, random_state=42, scoring='roc_auc'  # Reduced repeats for speed
        # )
        
        # # Create feature names
        # daily_feature_names = [f'Daily_Feature_{i}' for i in range(X_daily_features.shape[1])]
        # individual_feature_names = [f'Individual_Feature_{i}' for i in range(X_individual_features.shape[1])]
        # all_feature_names = daily_feature_names + individual_feature_names
           
    # feature_names = all_feature_names_all[0]  # Assumes all models use same feature names
    # importances_matrix = np.array([perm.importances_mean for perm in perm_importance_all])
    
    # # top 10 features per model
    # top10_lists = [list(np.array(feature_names)[np.argsort(imp)[-10:]][::-1]) for imp in importances_matrix]
    
    # # Check if all top 10 lists are the same
    # all_same = all(lst == top10_lists[0] for lst in top10_lists)
    
    # if all_same:
    #     top_features = top10_lists[0]
    # else:
    #     # Get top 11 from each model and count frequency
    #     top11_lists = [list(np.array(feature_names)[np.argsort(imp)[-11:]][::-1]) for imp in importances_matrix]
    #     flat_top11 = [feat for sublist in top11_lists for feat in sublist]
    #     most_common = Counter(flat_top11).most_common(10)
    #     top_features = [f for f, _ in most_common]
    
    # # Compute mean and std for the selected top features
    # indices = [feature_names.index(f) for f in top_features]
    # mean_importance = importances_matrix[:, indices].mean(axis=0)
    # std_importance = importances_matrix[:, indices].std(axis=0)
    
    # Average the predictions from all folds
    y_pred_target_agg /= num_splits
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    for threshold in thresholds:
        y_pred_binary = (y_pred_target_agg >= threshold).astype(int)
        
        if len(np.unique(y_pred_binary)) > 1:
            sens = recall_score(y_labels, y_pred_binary)
            spec = recall_score(y_labels, y_pred_binary, pos_label=0)
        else:
            sens = 0 if threshold > 0.5 else 1
            spec = 1 if threshold > 0.5 else 0
        
        sensitivities.append(sens)
        specificities.append(spec)
    
    # Convert to NumPy arrays for vectorized computation
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    # Compute absolute difference
    diff = np.abs(sensitivities - specificities)
    # Find indices where the difference is minimal
    min_diff = np.min(diff)
    candidates = np.where(diff == min_diff)[0]
    # Among those, find the one with highest average sensitivity + specificity
    best_idx = candidates[np.argmax((sensitivities[candidates] + specificities[candidates]) / 2)]
    best_threshold = thresholds[best_idx]
    
    y_pred_target_binary_agg = (y_pred_target_agg >= best_threshold).astype(int) 
    # Calculate prediction uncertainty (standard deviation across folds)
    prediction_std = np.std(all_predictions, axis=0)
    
    fpr, tpr, _ = roc_curve(y_labels, y_pred_target_agg)
    roc_auc = auc(fpr, tpr)
    
    # Calibration analysis
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_labels, y_pred_target_agg, n_bins=10
    )
    
    # Bootstrap confidence intervals for target dataset
    n_bootstraps = 1000
    rng = np.random.RandomState(42)
    aucs = []
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_labels), len(y_labels))
        if len(np.unique(y_labels[indices])) < 2:
            continue
        auc_bootstrap = roc_auc_score(y_labels[indices], y_pred_target_agg[indices])
        aucs.append(auc_bootstrap)
    
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)
    
    # Per-participant recall analysis
    df_results = pd.DataFrame({
        "participant_id": participant_ids.astype(int),
        "y_true": y_labels.flatten(),
        "y_pred_binary": y_pred_target_binary_agg,
        "y_pred_proba": y_pred_target_agg,
        "prediction_std": prediction_std
    })
    
    recall_per_participant = df_results.groupby("participant_id").apply(
        lambda g: recall_score(g["y_true"], g["y_pred_binary"]) if sum(g["y_true"]) > 0 else np.nan
    )

    return {
        "accuracy_scores": accuracy_score(y_labels, y_pred_target_binary_agg),
        "recall_scores": recall_score(y_labels, y_pred_target_binary_agg),
        "recall_scores_per_participant": recall_per_participant,
        "precision_scores": precision_score(y_labels, y_pred_target_binary_agg),
        "f1_scores": f1_score(y_labels, y_pred_target_binary_agg),
        "specificity_scores": recall_score(y_labels, y_pred_target_binary_agg, pos_label=0),
        "roc_auc_scores": roc_auc_score(y_labels, y_pred_target_agg),
        "fold_y_pred_proba": y_pred_target_agg,
        "roc_data": {"fpr": fpr, "tpr": tpr, "auc": roc_auc, "ci_lower": ci_lower, "ci_upper": ci_upper},
        "calibration_data": {"fraction_of_positives": fraction_of_positives, "mean_predicted_value": mean_predicted_value},
        # "feature_importance": {"feature_names": all_feature_names, "importance_mean": perm_importance.importances_mean,"importance_std": perm_importance.importances_std},
        "recall_per_participant": recall_per_participant,
        "conf_matrix": confusion_matrix(y_labels, y_pred_target_binary_agg)
    }

def cross_validation_holdout_analysis(label_name, num_splits=7, num_epochs=30, batch_s=32, num_patience=10, monitored='val_loss', threshold=0.3):
    configure_gpu()
    results = initialize_results()
    group_kfold = GroupKFold(n_splits=num_splits)
    file_path = 'home/datasets/adults_data.dill'
    X_matrices, X_daily_features, X_individual_features, labels, participant_ids = prepare_datasets(file_path)
    y_labels = labels[label_name]
    model_participants = np.unique(participant_ids)
    participant_labels = []
    participant_sample_indices = []
    
    for participant in model_participants:
        idx = np.where(participant_ids == participant)[0]
        participant_sample_indices.append(idx)
        label_values = y_labels[idx]
        participant_labels.append(int(np.any(label_values == 1)))  # binary: 1 if any day is positive
    
    participant_labels = np.array(participant_labels)
    sgkf = MultilabelStratifiedKFold(num_splits, shuffle=True, random_state=42)

    for fold, (train_val_group_idx, test_group_idx) in enumerate(
        sgkf.split(model_participants.reshape(-1, 1), 
                   np.vstack((1 - participant_labels, participant_labels)).T)
    ):            
        train_val_participants = model_participants[train_val_group_idx]
        test_participants = model_participants[test_group_idx]
    
        # Map back to sample indices
        train_val_idx = np.concatenate([np.where(participant_ids == p)[0] for p in train_val_participants])
        test_idx = np.concatenate([np.where(participant_ids == p)[0] for p in test_participants])

        train_idx, val_idx = get_train_val_test_indices(participant_ids, train_val_idx, test_idx, y_labels)
        success = False
        max_retries = 3
        for retry in range(max_retries):
            try:
                print(f"Processing fold {fold + 1} (Attempt {retry + 1}/{max_retries})...")
                
                # Clear memory before each training attempt
                tf.keras.backend.clear_session()
                gc.collect()
        
                model, model_history = train_and_evaluate_model(fold, train_idx, val_idx, X_matrices, 
                                                               X_daily_features, X_individual_features, y_labels, 
                                                               num_epochs, batch_s, num_patience, monitored)
        
                fold_results = evaluate_performance(model, test_idx, X_matrices, X_daily_features, 
                                                   X_individual_features, y_labels,participant_ids, threshold)
                
                for key in results.keys():
                    if key in fold_results.keys():
                        results[key].append(fold_results[key])
                results["fold_model"].append(model)
                results["fold_train_acc"].append(model_history.history['accuracy'])
                results["fold_train_auc"].append(model_history.history['AUC'])
                results["fold_train_loss"].append(model_history.history['loss'])
                results["fold_val_recall"].append(model_history.history['val_recall'])
                results["fold_val_acc"].append(model_history.history['val_accuracy'])
                results["fold_val_auc"].append(model_history.history['val_AUC'])
                results["fold_val_loss"].append(model_history.history['val_loss'])
        
                print(f"\nFold {fold + 1} completed successfully")
                # print(f"Fold {fold + 1} - {fold_results}")
                column_print = ["roc_auc_scores", "accuracy_scores", "recall_scores", "specificity_scores", "f1_scores"]
                print({k: v for k, v in fold_results.items() if k in column_print})
                print(f"\nROC-AUC - {fold_results['roc_data']['auc']}")
                success = True
                break
                
            except (tf.errors.InvalidArgumentError, tf.errors.ResourceExhaustedError, ValueError, RuntimeError) as e:
                print(f"Error in fold {fold + 1}, attempt {retry + 1}: {e}")
                tf.keras.backend.clear_session()
                gc.collect()
                time.sleep(10)  # Give system time to recover
        
        if not success:
            print(f"Failed to process fold {fold + 1} after {max_retries} attempts, skipping...")
            
        # Free GPU memory
        if 'model' in locals():
            del model
        if 'model_history' in locals():
            del model_history
        if 'fold_results' in locals():
            del fold_results
        gc.collect()
            
    # Calculate and print average and std across folds
    print(f'\nAverage Test Accuracy: {np.mean(results["accuracy_scores"]):.4f} (+- {np.std(results["accuracy_scores"]):.4f})')
    print(f'Average Test Recall: {np.mean(results["recall_scores"]):.4f} (+- {np.std(results["recall_scores"]):.4f})')
    print(f'Average Test Specificity: {np.mean(results["specificity_scores"]):.4f} (+- {np.std(results["specificity_scores"]):.4f})')
    print(f'Average Test AUC: {np.mean(results["roc_auc_scores"]):.4f} (+- {np.std(results["roc_auc_scores"]):.4f})')

    print(f"Aggregated Models on Pediatric set ... \n")
    file_path = 'home/datasets/pediatrics_data.dill'
    target_matrices, target_daily_features, target_individual_features, labels, target_participant_ids = prepare_datasets(file_path)
    target_labels = labels[label_name]
    
    target_daily_imputed, _, _ = median_impute_features(target_daily_features)
    target_daily_individual_imputed, _, _ = median_impute_features(target_individual_features)
    
    target_results = get_target_results(results["fold_model"], num_splits, target_matrices, 
                                       target_daily_imputed, target_daily_individual_imputed, 
                                       target_participant_ids, target_labels)

    print(f'Target Accuracy: {target_results["accuracy_scores"]:.4f}')
    print(f'Target Recall: {target_results["recall_scores"]:.4f}')
    print(f'Target Specificity: {target_results["specificity_scores"]:.4f}')
    print(f'Target AUC: {target_results["roc_auc_scores"]:.4f} \n')
    plot_training_history(results["fold_val_loss"], results["fold_val_auc"], results["fold_val_acc"], folder_name)

    plot_training_history_recall(results["fold_val_loss"], results["fold_val_auc"], results["fold_val_acc"], results["fold_val_recall"], folder_name)
    print(f'\nGenerating comprehensive validation plots and statistics...')
    
    # Feature Importance Plot
    # create_feature_importance_plot_from_results(results["feature_importance"], validation_folder)
    
    # Learning Curves (using existing function)
    # create_learning_curves_from_history(results["fold_val_loss"], results["fold_val_auc"], results["fold_val_acc"],
    #                                     results["fold_train_loss"], results["fold_train_auc"], results["fold_train_acc"], validation_folder)
    # Confusion Matrix and Performance Heatmap
    # create_confusion_matrix_and_performance_plot(results, validation_folder)
    
    # Statistical Significance Tests
    perform_statistical_tests(results, validation_folder)
    return results, target_results
    

