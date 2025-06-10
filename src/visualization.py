
"""
Visualization utilities for model results and analysis.
"""

def create_roc_comparison_plot_from_results(roc_data, save_folder):
    """Create ROC comparison plot from cross-validation results"""
    plt.figure(figsize=(6, 3))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    
    all_aucs = []
    all_ci_lowers = []
    all_ci_uppers = []
    
    for i, roc_data in enumerate(roc_data):
        plt.plot(roc_data['fpr'], roc_data['tpr'], color=colors[i], alpha=0.6, linewidth=1.5,
                label=f'Fold {i+1} (AUC = {roc_data["auc"]:.3f})')
        all_aucs.append(roc_data['auc'])
        all_ci_lowers.append(roc_data['ci_lower'])
        all_ci_uppers.append(roc_data['ci_upper'])
    
    # Average ROC curve
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_ci_lower = np.mean(all_ci_lowers)
    mean_ci_upper = np.mean(all_ci_uppers)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Random Classifier')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curves - Cross-Validation Results\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f} (95% CI: {mean_ci_lower:.3f}-{mean_ci_upper:.3f})', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'roc_curves_cv.png'), dpi=300, bbox_inches='tight')
    plt.show()
def create_feature_importance_plot_from_results(feature_importance, save_folder):
    """Create feature importance plot from cross-validation results"""
    if not feature_importance:
        return
    
    # Aggregate importance across folds
    all_importances = []
    feature_names = feature_importance[0]['feature_names']
    
    for fold_data in feature_importance:
        all_importances.append(fold_data['importance_mean'])
    
    avg_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(avg_importance)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(8, 6))
    y_pos = np.arange(len(sorted_idx))
    
    plt.barh(y_pos, avg_importance[sorted_idx], xerr=std_importance[sorted_idx],
             color=plt.cm.viridis(np.linspace(0, 1, len(sorted_idx))), alpha=0.8)
    
    plt.yticks(y_pos, [feature_names[i] for i in sorted_idx])
    plt.xlabel('Permutation Importance (AUC Decrease)', fontsize=14, fontweight='bold')
    plt.title('Top 20 Feature Importance\n(Average across Cross-Validation Folds)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, val, std) in enumerate(zip(sorted_idx, avg_importance[sorted_idx], std_importance[sorted_idx])):
        plt.text(val + std + 0.001, i, f'{val:.3f}±{std:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'feature_importance_cv.png'), dpi=300, bbox_inches='tight')
    gc.collect()
    plt.show()

def create_learning_curves_from_history(fold_val_loss, fold_val_auc, fold_val_acc,fold_train_loss, fold_train_auc, fold_train_acc, save_folder):
    """Create learning curves using existing logic"""
    max_epochs = min(len(fold) for fold in fold_val_auc) - 10
    # max_epochs = 100
    epochs_range = np.arange(1, max_epochs+1) 
    # Calculate average learning curves
    avg_loss = np.zeros(max_epochs)
    std_loss = np.zeros(max_epochs)
    avg_auc = np.zeros(max_epochs)
    std_auc = np.zeros(max_epochs)
    avg_acc = np.zeros(max_epochs)
    std_acc = np.zeros(max_epochs)
    
    train_avg_loss = np.zeros(max_epochs)
    train_std_loss = np.zeros(max_epochs)
    train_avg_auc = np.zeros(max_epochs)
    train_std_auc = np.zeros(max_epochs)
    train_avg_acc = np.zeros(max_epochs)
    train_std_acc = np.zeros(max_epochs)    
    for epoch in range(max_epochs):
        losses, aucs, accuracies, train_losses, train_aucs, train_accuracies = [], [], [], [], [], []
        for fold_loss, fold_auc, fold_acc,train_fold_loss, train_fold_auc, train_fold_acc in zip(fold_val_loss, fold_val_auc, fold_val_acc,fold_train_loss, fold_train_auc, fold_train_acc):
            if epoch < len(fold_loss):
                losses.append(fold_loss[epoch]/3)
                train_losses.append(train_fold_loss[epoch]/3)
            if epoch < len(fold_auc):
                aucs.append(fold_auc[epoch])
                train_aucs.append(train_fold_auc[epoch])
            if epoch < len(fold_acc):
                accuracies.append(fold_acc[epoch])
                train_accuracies.append(train_fold_acc[epoch])
        
        if losses:
            avg_loss[epoch] = np.mean(losses)
            std_loss[epoch] = np.std(losses)
            train_avg_loss[epoch] = np.mean(train_losses)
            train_std_loss[epoch] = np.std(train_losses)
        if aucs:
            avg_auc[epoch] = np.mean(aucs)
            std_auc[epoch] = np.std(aucs)
            train_avg_auc[epoch] = np.mean(train_aucs)
            train_std_auc[epoch] = np.std(train_aucs)
        if accuracies:
            avg_acc[epoch] = np.mean(accuracies)
            std_acc[epoch] = np.std(accuracies)
            train_avg_acc[epoch] = np.mean(train_accuracies)
            train_std_acc[epoch] = np.std(train_accuracies)
    
    plt.figure(figsize=(12, 3))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_avg_loss, color='#A23B72', linewidth=2.5, label=f'Training Loss', alpha=0.9)
    plt.fill_between(epochs_range, train_avg_loss - train_std_loss, train_avg_loss + train_std_loss, color='#A23B72', alpha=0.2)
    plt.plot(epochs_range, avg_loss, color='#A23B72', linewidth=2.5, linestyle='--', label=f'Validation Loss', alpha=0.9)
    plt.fill_between(epochs_range, avg_loss - std_loss, avg_loss + std_loss, color='#A23B72', alpha=0.15)
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Learning Curve: Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    # Add convergence annotation
    if len(avg_loss) > 30:
        converged_epoch = np.where(np.abs(np.diff(avg_loss[-10:])) < 0.005)[0]
        if len(converged_epoch) > 5:
            plt.axvline(x=epochs_range[-10], color='red', linestyle=':', alpha=0.7)
            plt.text(epochs_range[-10]+1, np.mean(avg_loss[-10:]), 'Converged', 
                   rotation=90, fontsize=11, color='red', fontweight='bold')
                
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_avg_auc, color='blue', linewidth=2.5, label=f'Training AUC-ROC', alpha=0.9)
    plt.fill_between(epochs_range, train_avg_auc - train_std_auc, train_avg_auc + train_std_auc, color='green', alpha=0.15)
    plt.plot(epochs_range, avg_auc, color='blue', linewidth=2.5, linestyle='--', label=f'Validation AUC-ROC', alpha=0.9)
    plt.fill_between(epochs_range, avg_auc - std_auc, avg_auc + std_auc, color='green', alpha=0.15)
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('AUC', fontsize=12, fontweight='bold')
    plt.title('Learning Curve: AUC', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)       
    plt.grid(True, alpha=0.3)
    # Add convergence annotation
    if len(avg_auc) > 30:
        converged_epoch = np.where(np.abs(np.diff(avg_auc[-10:])) < 0.005)[0]
        if len(converged_epoch) > 5:
            plt.axvline(x=epochs_range[-10], color='red', linestyle=':', alpha=0.7)
            plt.text(epochs_range[-10]+1, np.mean(avg_auc[-10:]), 'Converged', 
                   rotation=90, fontsize=11, color='red', fontweight='bold')
                
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_avg_acc, color='green', linewidth=2.5, label=f'Training Accuracy', alpha=0.9)
    plt.fill_between(epochs_range, train_avg_acc - train_std_acc, train_avg_acc + train_std_acc, color='green', alpha=0.15)
    plt.plot(epochs_range, avg_acc, color='green', linewidth=2.5, linestyle='--', label=f'Validation Accuracy', alpha=0.9)
    plt.fill_between(epochs_range, avg_acc - std_acc, avg_acc + std_acc, color='green', alpha=0.15)
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Learning Curve: Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    # Add convergence annotation
    if len(avg_acc) > 30:
        converged_epoch = np.where(np.abs(np.diff(avg_acc[-10:])) < 0.005)[0]
        if len(converged_epoch) > 5:
            plt.axvline(x=epochs_range[-10], color='red', linestyle=':', alpha=0.7)
            plt.text(epochs_range[-10]+1, np.mean(avg_acc[-10:]), 'Converged', 
                   rotation=90, fontsize=11, color='red', fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'learning_curves_enhanced.png'), dpi=300, bbox_inches='tight')
    gc.collect()
    plt.show()

def create_confusion_matrix_and_performance_plot(results, save_folder):
    """Create combined confusion matrix and performance metrics visualization"""
    fig = plt.figure(figsize=(6, 4))
    
    # Aggregate confusion matrices
    total_cm = np.sum(results["conf_matrix"], axis=0)
    
    # Create confusion matrix heatmap
    ax1 = plt.subplot(2, 2, (1, 2))
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar_kws={'label': 'Number of Samples'})
    ax1.set_title('Aggregated Confusion Matrix\n(All Cross-Validation Folds)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Performance metrics comparison table
    ax2 = plt.subplot(2, 2, 3)
    ax2.axis('tight')
    ax2.axis('off')
    
    metrics_data = [
        ['Metric', 'Mean ± Std', '95% CI'],
        ['Accuracy', f'{np.mean(results["accuracy_scores"]):.3f} ± {np.std(results["accuracy_scores"]):.3f}', 
         f'[{np.mean(results["accuracy_scores"]) - 1.96*np.std(results["accuracy_scores"]):.3f}, {np.mean(results["accuracy_scores"]) + 1.96*np.std(results["accuracy_scores"]):.3f}]'],
        ['Sensitivity', f'{np.mean(results["recall_scores"]):.3f} ± {np.std(results["recall_scores"]):.3f}',
         f'[{np.mean(results["recall_scores"]) - 1.96*np.std(results["recall_scores"]):.3f}, {np.mean(results["recall_scores"]) + 1.96*np.std(results["recall_scores"]):.3f}]'],
        ['Specificity', f'{np.mean(results["specificity_scores"]):.3f} ± {np.std(results["specificity_scores"]):.3f}',
         f'[{np.mean(results["specificity_scores"]) - 1.96*np.std(results["specificity_scores"]):.3f}, {np.mean(results["specificity_scores"]) + 1.96*np.std(results["specificity_scores"]):.3f}]'],
        ['AUC-ROC', f'{np.mean(results["roc_auc_scores"]):.3f} ± {np.std(results["roc_auc_scores"]):.3f}',
         f'[{np.mean(results["roc_auc_scores"]) - 1.96*np.std(results["roc_auc_scores"]):.3f}, {np.mean(results["roc_auc_scores"]) + 1.96*np.std(results["roc_auc_scores"]):.3f}]']
    ]
    
    table = ax2.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Header styling
    for i in range(len(metrics_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        for j in range(len(metrics_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax2.set_title('Cross-Validation Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    
    # Box plot of performance metrics
    ax3 = plt.subplot(2, 2, 4)
    metrics_for_boxplot = [
        results["accuracy_scores"],
        results["recall_scores"], 
        results["specificity_scores"],
        results["precision_scores"],
        results["f1_scores"],
        results["roc_auc_scores"]
    ]
    labels = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1-Score', 'AUC-ROC']
    
    bp = ax3.boxplot(metrics_for_boxplot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Performance Distribution Across Folds', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'confusion_matrix_and_performance.png'), dpi=300, bbox_inches='tight')
    gc.collect()
    plt.show()


def create_calibration_plots_from_results(calibration_data, save_folder):
    """Create calibration plots from cross-validation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(calibration_data)))
    
    # Individual fold calibration curves
    all_fractions = []
    all_means = []
    
    for i, cal_data in enumerate(calibration_data):
        ax1.plot(cal_data['mean_predicted_value'], cal_data['fraction_of_positives'], 
                'o-', color=colors[i], alpha=0.7, label=f'Fold {i+1}', markersize=3, linewidth=2)
        all_fractions.append(cal_data['fraction_of_positives'])
        all_means.append(cal_data['mean_predicted_value'])
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
    ax1.set_title('Individual Fold Calibration', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Average calibration curve
    if all_fractions:
        avg_fractions = np.mean(all_fractions, axis=0)
        avg_means = np.mean(all_means, axis=0)
        std_fractions = np.std(all_fractions, axis=0)
        
        ax2.plot(avg_means, avg_fractions, 'bo-', linewidth=2, markersize=3, label='Average Calibration')
        ax2.fill_between(avg_means, avg_fractions - std_fractions, avg_fractions + std_fractions, 
                        alpha=0.3, color='blue')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')

        ax2.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
        ax2.set_title('Average Calibration Curve', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'calibration_panel_6h.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'calibration_panel_6h.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
def create_calibration_panel(results, target_results, save_folder):
    """Create calibration plots from cross-validation results"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 4))

    fold_y_true = results["fold_y_true"]
    fold_y_pred_proba = results["fold_y_pred_proba"]
    current_sens = np.mean(results["recall_scores"])
    current_spec = np.mean(results["specificity_scores"])
    all_y_true = np.concatenate(fold_y_true)
    all_y_pred_proba = np.concatenate(fold_y_pred_proba)
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    for threshold in thresholds:
        y_pred_binary = (all_y_pred_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred_binary)) > 1:
            sens = recall_score(all_y_true, y_pred_binary)
            spec = recall_score(all_y_true, y_pred_binary, pos_label=0)
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
    print(best_threshold)
    current_threshold = 0.32 #################################### current_threshold  ####################################
    ax1.plot(thresholds, sensitivities, 'g-', linewidth=2, label='Sensitivity')
    ax1.plot(thresholds, specificities, 'b-', linewidth=2, label='Specificity')
    ax1.axvline(x=current_threshold, color='r', linestyle='--', alpha=0.7, label=f'Current Threshold ({current_threshold:.2f})')
    ax1.axvline(x=best_threshold, color='b', linestyle='--', alpha=0.7, label=f'Optimal Threshold ({best_threshold:.2f})')
    ax1.annotate(f'Optimized Performance\nSens={current_sens:.2f}, Spec={current_spec:.2f}', 
                xy=(best_threshold, (current_sens + current_spec)/2), 
                xytext=(best_threshold + 0.08, 0.6),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, #fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='lightyellow', alpha=0.8))
    
    ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('A. Sensitivity vs Specificity Trade-off\nAdult Cohort', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower center',fontsize=10)
    ax1.grid(True, alpha=0.3)
    label_name = "hypo_day"
    file_path = PATH
    target_matrices, target_daily_features, target_individual_features, labels, target_participant_ids = prepare_datasets(file_path)
    all_y_true = labels[label_name]
    
    all_y_pred_proba = target_results["fold_y_pred_proba"]
    current_sens = np.mean(target_results["recall_scores"])
    current_spec = np.mean(target_results["specificity_scores"])
    print(np.shape(all_y_pred_proba))
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    for threshold in thresholds:
        y_pred_binary = (all_y_pred_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred_binary)) > 1:
            sens = recall_score(all_y_true, y_pred_binary)
            spec = recall_score(all_y_true, y_pred_binary, pos_label=0)
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
    print(best_threshold)
    current_threshold = 0.23 #################################### current_threshold  ####################################
    ax2.plot(thresholds, sensitivities, 'g-', linewidth=2, label='Sensitivity')
    ax2.plot(thresholds, specificities, 'b-', linewidth=2, label='Specificity')
    ax2.axvline(x=current_threshold, color='r', linestyle='--', alpha=0.7, label=f'Current Threshold ({current_threshold:.2f})')
    ax2.axvline(x=best_threshold, color='b', linestyle='--', alpha=0.7, label=f'Optimal Threshold ({best_threshold:.2f})')
    ax2.annotate(f'Optimized Performance\nSens={current_sens:.2f}, Spec={current_spec:.2f}', 
                xy=(best_threshold, (current_sens + current_spec)/2), 
                xytext=(best_threshold + 0.08, 0.6),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, #fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.1", facecolor='lightyellow', alpha=0.8))
    
    ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    # ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax2.set_title('B. Sensitivity vs Specificity Trade-off\nPediatric Cohort', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower center', fontsize=10)
    ax2.grid(True, alpha=0.3)

    calibration_data = results["calibration_data"] 
    target_calibration_data = target_results["calibration_data"]
    all_fractions = []
    all_means = []
    for i, cal_data in enumerate(calibration_data):
        all_fractions.append(cal_data['fraction_of_positives'])
        all_means.append(cal_data['mean_predicted_value'])
    avg_fractions = np.mean(all_fractions, axis=0)
    avg_means = np.mean(all_means, axis=0)
    std_fractions = np.std(all_fractions, axis=0)
    
    ax3.plot(avg_means, avg_fractions, 'bo-', linewidth=2, markersize=2, label='Adult Cohort\nAverage Calibration', color='blue')
    ax3.fill_between(avg_means, avg_fractions - std_fractions, avg_fractions + std_fractions, 
                    alpha=0.3, color='blue')
    ax3.plot(target_calibration_data['mean_predicted_value'], target_calibration_data['fraction_of_positives'], 'bo-', 
             linewidth=2, markersize=2, label='Pediatric Cohort\nCalibration', color='purple')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')

    ax3.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax3.set_title('C. Pediatric Data Calibration Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left',fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'full_calibration_panel_6h.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'full_calibration_panel_6h.pdf'), dpi=300, bbox_inches='tight')
    gc.collect()
    plt.show()
    

def plot_training_history_recall(folds_val_loss, folds_val_auc, folds_val_acc, folds_val_rec, folder):
    # Determine max_epochs as the minimum length of the folds minus 10
    max_epochs = min(len(fold) for fold in folds_val_auc) - 2

    # Initialize arrays to store average and standard deviation for each epoch
    avg_loss = np.zeros(max_epochs)
    std_loss = np.zeros(max_epochs)
    avg_auc = np.zeros(max_epochs)
    std_auc = np.zeros(max_epochs)
    avg_acc = np.zeros(max_epochs)
    std_acc = np.zeros(max_epochs)
    avg_rec = np.zeros(max_epochs)
    std_rec = np.zeros(max_epochs)
    
    # Compute average and standard deviation for each epoch
    for epoch in range(max_epochs):
        losses = []
        aucs = []
        accuracies = []
        recalls = []
        for fold_loss, fold_auc, fold_acc, fold_rec in zip(folds_val_loss, folds_val_auc, folds_val_acc, folds_val_rec):
            if epoch < len(fold_loss):
                losses.append(fold_loss[epoch]/3)
            if epoch < len(fold_auc):
                aucs.append(fold_auc[epoch])
            if epoch < len(fold_acc):
                accuracies.append(fold_acc[epoch])
            if epoch < len(fold_rec):
                recalls.append(fold_rec[epoch])
        if losses:
            avg_loss[epoch] = np.mean(losses)
            std_loss[epoch] = np.std(losses)
        if aucs:
            avg_auc[epoch] = np.mean(aucs)
            std_auc[epoch] = np.std(aucs)
        if accuracies:
            avg_acc[epoch] = np.mean(accuracies)
            std_acc[epoch] = np.std(accuracies)
        if accuracies:
            avg_rec[epoch] = np.mean(recalls)
            std_rec[epoch] = np.std(recalls)
            
    # Adjusted x-axis range: 1 to 3*max_epochs + 1 with step size of 3
    epochs_range = range(1, 3 * max_epochs + 1, 3)

    # Plot average loss, AUC, and accuracy
    plt.figure(figsize=(10, 8))

    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, avg_loss, color='blue', label='Validation Loss')
    plt.fill_between(epochs_range, 
                     avg_loss - std_loss,
                     avg_loss + std_loss,
                     color='blue', alpha=0.2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Validation Loss', fontsize=12)
    plt.legend(fontsize=10)

    # AUC plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, avg_auc, color='blue', label='Validation AUC')
    plt.fill_between(epochs_range, 
                     avg_auc - std_auc,
                     avg_auc + std_auc,
                     color='blue', alpha=0.2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('AUC', fontsize=14)
    plt.title('Validation AUC', fontsize=12)
    plt.legend(fontsize=10)

    # Accuracy plot
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, avg_acc, color='blue', label='Validation Accuracy')
    plt.fill_between(epochs_range, 
                     avg_acc - std_acc,
                     avg_acc + std_acc,
                     color='blue', alpha=0.2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Validation Accuracy', fontsize=12)
    plt.legend(fontsize=10)

    # Accuracy plot
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, avg_rec, color='blue', label='Validation Sensitivity')
    plt.fill_between(epochs_range, 
                     avg_rec - std_rec,
                     avg_rec + std_rec,
                     color='blue', alpha=0.2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Sensitivity', fontsize=14)
    plt.title('Validation Sensitivity', fontsize=12)
    plt.legend(fontsize=10)
    
    # Tight layout and save
    plt.tight_layout()
    plot_path = os.path.join(folder, 'loss_auc_accuracy_recall_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")
    plt.show()
    
def plot_training_history(folds_val_loss, folds_val_auc, folds_val_acc, folder):
    # Determine max_epochs as the minimum length of the folds minus 10
    max_epochs = min(len(fold) for fold in folds_val_auc) - 2

    # Initialize arrays to store average and standard deviation for each epoch
    avg_loss = np.zeros(max_epochs)
    std_loss = np.zeros(max_epochs)
    avg_auc = np.zeros(max_epochs)
    std_auc = np.zeros(max_epochs)
    avg_acc = np.zeros(max_epochs)
    std_acc = np.zeros(max_epochs)

    # Compute average and standard deviation for each epoch
    for epoch in range(max_epochs):
        losses = []
        aucs = []
        accuracies = []
        for fold_loss, fold_auc, fold_acc in zip(folds_val_loss, folds_val_auc, folds_val_acc):
            if epoch < len(fold_loss):
                losses.append(fold_loss[epoch]/3)
            if epoch < len(fold_auc):
                aucs.append(fold_auc[epoch])
            if epoch < len(fold_acc):
                accuracies.append(fold_acc[epoch])
        if losses:
            avg_loss[epoch] = np.mean(losses)
            std_loss[epoch] = np.std(losses)
        if aucs:
            avg_auc[epoch] = np.mean(aucs)
            std_auc[epoch] = np.std(aucs)
        if accuracies:
            avg_acc[epoch] = np.mean(accuracies)
            std_acc[epoch] = np.std(accuracies)

    # Adjusted x-axis range: 1 to 3*max_epochs + 1 with step size of 3
    epochs_range = range(1, 3 * max_epochs + 1, 3)

    # Plot average loss, AUC, and accuracy
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, avg_loss, color='blue', label='Validation Loss')
    plt.fill_between(epochs_range, 
                     avg_loss - std_loss,
                     avg_loss + std_loss,
                     color='blue', alpha=0.2)
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Validation Loss per Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14)

    # AUC plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, avg_auc, color='blue', label='Validation AUC')
    plt.fill_between(epochs_range, 
                     avg_auc - std_auc,
                     avg_auc + std_auc,
                     color='blue', alpha=0.2)
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('AUC', fontsize=14, fontweight='bold')
    plt.title('Validation AUC per Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)

    # Accuracy plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, avg_acc, color='blue', label='Validation Accuracy')
    plt.fill_between(epochs_range, 
                     avg_acc - std_acc,
                     avg_acc + std_acc,
                     color='blue', alpha=0.2)
    plt.xlabel('Epochs', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Validation Accuracy per Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=14)

    # Tight layout and save
    plt.tight_layout()
    plot_path = os.path.join(folder, 'loss_auc_accuracy_plot.png')
    plt.savefig(plot_path)
    print(f"Plot saved at: {plot_path}")
    plt.show()


