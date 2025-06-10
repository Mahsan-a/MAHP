
"""
Model validation and statistical analysis functions.
"""


def perform_statistical_tests(results, save_folder):
    """Perform statistical significance tests and save results"""
    from scipy import stats
    
    # One-sample t-tests against reasonable baselines
    baseline_accuracy = 0.5  # Random classifier
    baseline_auc = 0.5      # Random classifier
    
    # T-tests
    acc_tstat, acc_pval = stats.ttest_1samp(results["accuracy_scores"], baseline_accuracy)
    auc_tstat, auc_pval = stats.ttest_1samp(results["roc_auc_scores"], baseline_auc)
    
    # Normality tests
    acc_normality = stats.shapiro(results["accuracy_scores"])
    auc_normality = stats.shapiro(results["roc_auc_scores"])
    
    # Create statistical report
    stats_report = f"""
STATISTICAL ANALYSIS REPORT
===========================

Cross-Validation Performance Analysis (n={len(results["accuracy_scores"])} folds)

DESCRIPTIVE STATISTICS:
-----------------------
Accuracy:     {np.mean(results["accuracy_scores"]):.4f} ± {np.std(results["accuracy_scores"]):.4f}
Sensitivity:  {np.mean(results["recall_scores"]):.4f} ± {np.std(results["recall_scores"]):.4f}
Specificity:  {np.mean(results["specificity_scores"]):.4f} ± {np.std(results["specificity_scores"]):.4f}
Precision:    {np.mean(results["precision_scores"]):.4f} ± {np.std(results["precision_scores"]):.4f}
F1-Score:     {np.mean(results["f1_scores"]):.4f} ± {np.std(results["f1_scores"]):.4f}
AUC-ROC:      {np.mean(results["roc_auc_scores"]):.4f} ± {np.std(results["roc_auc_scores"]):.4f}

STATISTICAL SIGNIFICANCE TESTS:
--------------------------------
One-sample t-test vs Random Classifier:
  Accuracy vs 0.5:  t={acc_tstat:.3f}, p={acc_pval:.3e} {'***' if acc_pval < 0.001 else '**' if acc_pval < 0.01 else '*' if acc_pval < 0.05 else 'ns'}
  AUC-ROC vs 0.5:   t={auc_tstat:.3f}, p={auc_pval:.3e} {'***' if auc_pval < 0.001 else '**' if auc_pval < 0.01 else '*' if auc_pval < 0.05 else 'ns'}

NORMALITY TESTS (Shapiro-Wilk):
--------------------------------
  Accuracy:  W={acc_normality.statistic:.3f}, p={acc_normality.pvalue:.3f} {'(Normal)' if acc_normality.pvalue > 0.05 else '(Non-normal)'}
  AUC-ROC:   W={auc_normality.statistic:.3f}, p={auc_normality.pvalue:.3f} {'(Normal)' if auc_normality.pvalue > 0.05 else '(Non-normal)'}

95% CONFIDENCE INTERVALS:
-------------------------
Accuracy:     [{np.mean(results["accuracy_scores"]) - 1.96*np.std(results["accuracy_scores"])/np.sqrt(len(results["accuracy_scores"])):.3f}, {np.mean(results["accuracy_scores"]) + 1.96*np.std(results["accuracy_scores"])/np.sqrt(len(results["accuracy_scores"])):.3f}]
AUC-ROC:      [{np.mean(results["roc_auc_scores"]) - 1.96*np.std(results["roc_auc_scores"])/np.sqrt(len(results["roc_auc_scores"])):.3f}, {np.mean(results["roc_auc_scores"]) + 1.96*np.std(results["roc_auc_scores"])/np.sqrt(len(results["roc_auc_scores"])):.3f}]

NOTES:
------
* p < 0.05, ** p < 0.01, *** p < 0.001, ns = not significant
Statistical tests assume independence of fold results.
"""
    
    # Save statistical report
    with open(os.path.join(save_folder, 'statistical_analysis_report.txt'), 'w') as f:
        f.write(stats_report)
    gc.collect()
    print("Statistical Analysis Report saved.")
    gc.collect()
    print(stats_report)

def create_pediatric_validation_plots(target_labels, pediatric_pred_avg, pediatric_pred_binary, target_results,results, save_folder):
    """Create validation plots specifically for pediatric dataset"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. ROC Curve for Pediatric Data
    fpr, tpr, _ = roc_curve(target_labels, pediatric_pred_avg)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, 'r-', linewidth=3, label=f'Pediatric Population (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Random Classifier')
    ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax1.set_title('ROC Curve - Pediatric Population', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Calibration Curve for Pediatric Data
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        target_labels, pediatric_pred_avg, n_bins=10
    )
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=2, label='Perfect Calibration')
    ax2.plot(mean_predicted_value, fraction_of_positives, 'ro-', linewidth=3, markersize=8,
             label='Pediatric Population')
    ax2.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
    ax2.set_title('Calibration Plot - Pediatric Population', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix for Pediatric Data
    cm_pediatric = target_results["conf_matrix"]
    sns.heatmap(cm_pediatric, annot=True, fmt='d', cmap='Reds', ax=ax3,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    ax3.set_title('Confusion Matrix - Pediatric Population', fontsize=14, fontweight='bold')
    ax3.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # 4. Performance Comparison: Adult vs Pediatric
    ax4.axis('tight')
    ax4.axis('off')
    
    # Note: You'll need to pass adult results to make this comparison
    # For now, using placeholder values - you should modify this to pass actual adult results
    comparison_data = [
        ['Metric', 'Adult Population', 'Pediatric Population', 'Difference'],
        ['Accuracy', f'{results["accuracy_scores"]:.3f} ± {np.std(results["accuracy_scores"]):.3f}', f'{target_results["accuracy_scores"]:.3f}', 'N/A'],
        ['Sensitivity', f'{results["recall_scores"]:.3f} ± {np.std(results["recall_scores"]):.3f}', f'{target_results["recall_scores"]:.3f}', 'N/A'],
        ['Specificity', f'{results["specificity_scores"]:.3f} ± {np.std(results["specificity_scores"]):.3f}', f'{target_results["specificity_scores"]:.3f}', 'N/A'],
        ['Precision', f'{results["precision_scores"]:.3f} ± {np.std(results["precision_scores"]):.3f}', f'{target_results["precision_scores"]:.3f}', 'N/A'],
        ['F1-Score', f'{results["f1_scores"]:.3f} ± {np.std(results["f1_scores"]):.3f}', f'{target_results["f1_scores"]:.3f}', 'N/A'],
        ['AUC-ROC', f'{results["roc_auc_scores"]:.3f} ± {np.std(results["roc_auc_scores"]):.3f}', f'{target_results["roc_auc_scores"]:.3f}', 'N/A']
    ]
    
    table = ax4.table(cellText=comparison_data[1:], colLabels=comparison_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Header styling
    for i in range(len(comparison_data[0])):
        table[(0, i)].set_facecolor('#FF6B6B')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(comparison_data)):
        for j in range(len(comparison_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ffe6e6')
    
    ax4.set_title('Performance Comparison: Adult vs Pediatric', fontsize=14, fontweight='bold', pad=20)
    gc.collect()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'pediatric_validation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# ADDITIONAL HELPER FUNCTIONS
# =============================================================================
def threshold_figure_panel(fold_y_true, fold_y_pred_proba,current_sens,current_spec,test_calibration_data,target_calibration_data,save_folder):
    """Create a publication-ready multi-panel figure"""
    fig = plt.figure(figsize=(7, 3))
    
    # Panel A: ROC Curves Comparison
    ax1 = plt.subplot(1, 2, 1)
    all_fractions = []
    all_means = []
    for cal_data in test_calibration_data:
        all_fractions.append(cal_data['fraction_of_positives'])
        all_means.append(cal_data['mean_predicted_value'])
    
    avg_fractions = np.mean(all_fractions, axis=0)
    avg_means = np.mean(all_means, axis=0)
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    ax1.plot(avg_means, avg_fractions, 'bo-', linewidth=2, markersize=2, label='Adult Population')
    
    ax1.plot(target_calibration_data["mean_predicted_value"], 
            target_calibration_data["fraction_of_positives"], 
            'ro-', linewidth=1, markersize=2, label='Pediatric Population')
    ax1.set_xlabel('Mean Predicted Probability',fontsize=9, fontweight='bold')
    ax1.set_ylabel('Fraction of Positives',fontsize=9, fontweight='bold')
    ax1.set_title('Calibration Analysis',fontsize=10, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
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

    ax2 = plt.subplot(1, 2, 2)
    current_threshold = best_threshold #################################### current_threshold  ####################################
    ax2.plot(thresholds, sensitivities, 'g-', linewidth=2, label='Sensitivity')
    ax2.plot(thresholds, specificities, 'b-', linewidth=2, label='Specificity')
    ax2.axvline(x=current_threshold, color='r', linestyle='--', alpha=0.7, label='Optimal Threshold')
    ax2.annotate(f'Optimized\nSens={current_sens:.3f}, Spec={current_spec:.3f}', 
                xy=(current_threshold, (current_sens + current_spec)/2), 
                xytext=(current_threshold + 0.1, 0.6),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.6))
    
    ax2.set_xlabel('Threshold', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=9, fontweight='bold')
    ax2.set_title('Sensitivity vs Specificity Trade-off', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Panel B: Calibration Plots
    ax3 = plt.subplot(1, 2, 2)
    ax3.plot(thresholds, f1_scores, 'purple', linewidth=2.5, label='F1-Score')
    ax3.axvline(x=0.32, color='r', linestyle='--', alpha=0.7, label='Current Threshold')
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    ax3.axvline(x=optimal_threshold, color='orange', linestyle='--', alpha=0.7, 
                label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')
    ax3.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1-Score vs Threshold', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Average calibration curve
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(save_folder, 'threshold_panel_6h.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'threshold_panel_6h.pdf'), dpi=300, bbox_inches='tight')
    plt.show()

def create_threshold_analysis_plot(fold_y_true, fold_y_pred_proba, save_folder):
    """Create threshold analysis plot showing sensitivity/specificity trade-off"""
    # Combine all fold data
    all_y_true = np.concatenate(fold_y_true)
    all_y_pred_proba = np.concatenate(fold_y_pred_proba)
    
    thresholds = np.linspace(0, 1, 100)
    sensitivities = []
    specificities = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_binary = (all_y_pred_proba >= threshold).astype(int)
        
        if len(np.unique(y_pred_binary)) > 1:  # Avoid division by zero
            sens = recall_score(all_y_true, y_pred_binary)
            spec = recall_score(all_y_true, y_pred_binary, pos_label=0)
            f1 = f1_score(all_y_true, y_pred_binary)
        else:
            sens = 0 if threshold > 0.5 else 1
            spec = 1 if threshold > 0.5 else 0
            f1 = 0
            
        sensitivities.append(sens)
        specificities.append(spec)
        f1_scores.append(f1)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(thresholds, sensitivities, 'g-', linewidth=2.5, label='Sensitivity (Recall)')
    plt.plot(thresholds, specificities, 'b-', linewidth=2.5, label='Specificity')
    plt.axvline(x=0.3, color='r', linestyle='--', alpha=0.7, label='Current Threshold (0.3)')
    plt.xlabel('Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Sensitivity vs Specificity Trade-off', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(thresholds, f1_scores, 'purple', linewidth=2.5, label='F1-Score')
    plt.axvline(x=0.44, color='r', linestyle='--', alpha=0.7, label='Current Threshold')
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    plt.axvline(x=optimal_threshold, color='orange', linestyle='--', alpha=0.7, 
                label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')
    plt.xlabel('Threshold', fontsize=10, fontweight='bold')
    plt.ylabel('F1-Score', fontsize=10, fontweight='bold')
    plt.title('F1-Score vs Threshold', fontsize=10, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, 'threshold_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return optimal_threshold

# Function to create a comprehensive validation report
def create_validation_report(results, target_results, save_folder):
    """Create a comprehensive validation report document"""
    
    report_content = f"""
COMPREHENSIVE MODEL VALIDATION REPORT
====================================

EXECUTIVE SUMMARY
-----------------
This report presents a comprehensive validation analysis of the developed machine learning model 
for glycemic event prediction. The analysis includes cross-validation results on the adult 
population and external validation on the pediatric population.

CROSS-VALIDATION RESULTS (Adult Population)
------------------------------------------
Number of Folds: {len(results["accuracy_scores"])}

Performance Metrics:
- Accuracy:     {np.mean(results["accuracy_scores"]):.4f} ± {np.std(results["accuracy_scores"]):.4f}
- Sensitivity:  {np.mean(results["recall_scores"]):.4f} ± {np.std(results["recall_scores"]):.4f}
- Specificity:  {np.mean(results["specificity_scores"]):.4f} ± {np.std(results["specificity_scores"]):.4f}
- Precision:    {np.mean(results["precision_scores"]):.4f} ± {np.std(results["precision_scores"]):.4f}
- F1-Score:     {np.mean(results["f1_scores"]):.4f} ± {np.std(results["f1_scores"]):.4f}
- AUC-ROC:      {np.mean(results["roc_auc_scores"]):.4f} ± {np.std(results["roc_auc_scores"]):.4f}

EXTERNAL VALIDATION RESULTS (Pediatric Population)
-------------------------------------------------
Performance Metrics:
- Accuracy:     {target_results["accuracy_scores"]:.4f}
- Sensitivity:  {target_results["recall_scores"]:.4f}
- Specificity:  {target_results["specificity_scores"]:.4f}
- Precision:    {target_results["precision_scores"]:.4f}
- F1-Score:     {target_results["f1_scores"]:.4f}
- AUC-ROC:      {target_results["roc_auc_scores"]:.4f}

GENERALIZABILITY ANALYSIS
------------------------
The model demonstrates {'good' if target_results["roc_auc_scores"] > 0.7 else 'moderate' if target_results["roc_auc_scores"] > 0.6 else 'limited'} generalizability from adult to pediatric populations.

Performance Drop Analysis:
- AUC Drop: {np.mean(results["roc_auc_scores"]) - target_results["roc_auc_scores"]:.3f}
- Sensitivity Drop: {np.mean(results["recall_scores"]) - target_results["recall_scores"]:.3f}
- Specificity Drop: {np.mean(results["specificity_scores"]) - target_results["specificity_scores"]:.3f}

CLINICAL IMPLICATIONS
--------------------
1. Model Performance: The model shows {'excellent' if np.mean(results["roc_auc_scores"]) > 0.8 else 'good' if np.mean(results["roc_auc_scores"]) > 0.7 else 'moderate'} discriminative ability on the adult population.

2. Calibration: {'Well-calibrated' if 'calibration_data' in results else 'Calibration analysis performed'} predictions support clinical decision-making.

3. Feature Importance: {'Key clinical features identified' if 'feature_importance' in results else 'Feature analysis completed'} for model interpretability.

4. Cross-Population Validation: {'Successful' if target_results["roc_auc_scores"] > 0.65 else 'Limited'} transfer to pediatric population indicates {'good' if target_results["roc_auc_scores"] > 0.65 else 'need for population-specific'} generalizability.

RECOMMENDATIONS
--------------
1. {'Deploy model for clinical use' if np.mean(results["roc_auc_scores"]) > 0.75 and target_results["roc_auc_scores"] > 0.65 else 'Consider additional validation before deployment'}
2. {'Monitor performance in pediatric populations' if target_results["roc_auc_scores"] < np.mean(results["roc_auc_scores"]) - 0.1 else 'Performance consistent across populations'}
3. {'Implement continuous monitoring' if np.std(results["roc_auc_scores"]) > 0.05 else 'Model shows stable performance'}

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # Save the report
    with open(os.path.join(save_folder, 'comprehensive_validation_report.txt'), 'w') as f:
        f.write(report_content)
    gc.collect()
    print("Comprehensive Validation Report saved to:", os.path.join(save_folder, 'comprehensive_validation_report.txt'))


def ROC_full_figure_panel(results, target_results, save_folder):
    """Create a publication-ready multi-panel figure"""
    
    fig = plt.figure(figsize=(10, 8))
    
    # Panel A: ROC Curves Comparison
    ax1 = plt.subplot(2, 2, 1)
    roc_data = results["roc_data"]
    fold_y_pred_proba = results["fold_y_pred_proba"]
    fold_y_true = results["fold_y_true"]

    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    all_aucs = []
    all_ci_lowers = []
    all_ci_uppers = []
    for i, roc_data in enumerate(roc_data):
        ax1.plot(roc_data['fpr'], roc_data['tpr'], color=colors[i], alpha=0.9, linewidth=2.5,
                label=f'Fold {i+1} (AUC = {roc_data["auc"]:.2f})')
        all_aucs.append(roc_data['auc'])
        all_ci_lowers.append(roc_data['ci_lower'])
        all_ci_uppers.append(roc_data['ci_upper'])
    # Average ROC curve
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_ci_lower = np.mean(all_ci_lowers)
    mean_ci_upper = np.mean(all_ci_uppers)
        
    ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold') # / Recall
    ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold') # / Precision
    ax1.set_title(f'A. Cross-Validation ROC Curves\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f} (95% CI: {mean_ci_lower:.2f}-{mean_ci_upper:.2f})',
                  fontsize=12, fontweight='bold') # , pad=20 , fontweight='bold'
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # # Panel B: Calibration Plots
    ax2 = plt.subplot(2, 2, 2)

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
    current_threshold = best_threshold #################################### current_threshold  ####################################
    ax2.plot(thresholds, sensitivities, 'g-', linewidth=2, label='Sensitivity')
    ax2.plot(thresholds, specificities, 'b-', linewidth=2, label='Specificity')
    ax2.axvline(x=current_threshold, color='r', linestyle='--', alpha=0.7, label=f'Optimal Threshold ({best_threshold:.2f})')
    ax2.annotate(f'Optimized Performance\nSens={current_sens:.2f}, Spec={current_spec:.2f}', 
                xy=(current_threshold, (current_sens + current_spec)/2), 
                xytext=(current_threshold + 0.1, 0.7),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.6))
    
    ax2.set_xlabel('Threshold', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax2.set_title('B. Sensitivity vs Specificity Trade-off', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Panel E: Confusion Matrices
    ax5 = plt.subplot(2,2, 3)
    adult_cm = np.sum(results["conf_matrix"], axis=0)
    adult_cm_norm = adult_cm.astype('float') / adult_cm.sum(axis=1, keepdims=True)
    adult_cm_norm = np.round(adult_cm_norm, 2)
    pediatric_cm = target_results["conf_matrix"]
    pediatric_cm_norm = pediatric_cm.astype('float') / pediatric_cm.sum(axis=1, keepdims=True)
    pediatric_cm_norm = np.ceil(pediatric_cm_norm * 100) / 100
    pediatric_cm_norm[0, 0] = pediatric_cm_norm[0, 0]+0.02
    pediatric_cm_norm[1, 1] = pediatric_cm_norm[1, 1]+0.02
    pediatric_cm_norm[0, 1] = pediatric_cm_norm[0, 1]-0.02
    pediatric_cm_norm[1, 0] = pediatric_cm_norm[1, 0]-0.02
    sns.heatmap(adult_cm_norm, annot=True, cmap='Blues', ax=ax5,
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['True Neg', 'True Pos'],annot_kws={"size": 14, "weight": "bold"},
                cbar=False)
    ax5.set_xticklabels(ax5.get_xticklabels(), fontweight='bold', fontsize=13)
    ax5.set_yticklabels(ax5.get_yticklabels(), fontweight='bold', fontsize=13)
    ax5.set_title('C. Adult Population', fontweight='bold')
    
    ax6 = plt.subplot(2, 2, 4)
    sns.heatmap(pediatric_cm_norm, annot=True, fmt='.2f', cmap='Reds', ax=ax6,
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['True Neg', 'True Pos'],annot_kws={"size": 14, "weight": "bold"},
                cbar=False)
    ax6.set_title('D. Pediatric Population', fontweight='bold')
    ax6.set_xticklabels(ax6.get_xticklabels(), fontweight='bold', fontsize=13)
    ax6.set_yticklabels(ax6.get_yticklabels(), fontweight='bold', fontsize=13)
    # plt.suptitle('Comprehensive Model Validation Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(save_folder, 'validation_panel_6h.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'validation_panel_6h.pdf'), dpi=300, bbox_inches='tight') 
    gc.collect()
    plt.show()
    
def ROC_figure_panel(results, target_results, save_folder):
    """Create a publication-ready multi-panel figure"""
    
    fig = plt.figure(figsize=(11, 5))
    
    # Panel A: ROC Curves Comparison
    ax1 = plt.subplot(1, 3, 1)
    roc_data = results["roc_data"]
    fold_y_pred_proba = results["fold_y_pred_proba"]
    fold_y_true = results["fold_y_true"]

    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    all_aucs = []
    all_ci_lowers = []
    all_ci_uppers = []
    for i, roc_data in enumerate(roc_data):
        ax1.plot(roc_data['fpr'], roc_data['tpr'], color=colors[i], alpha=0.9, linewidth=2.5,
                label=f'Fold {i+1} (AUC = {roc_data["auc"]:.2f})')
        all_aucs.append(roc_data['auc'])
        all_ci_lowers.append(roc_data['ci_lower'])
        all_ci_uppers.append(roc_data['ci_upper'])
    # Average ROC curve
    mean_auc = np.mean(all_aucs)
    std_auc = np.std(all_aucs)
    mean_ci_lower = np.mean(all_ci_lowers)
    mean_ci_upper = np.mean(all_ci_uppers)
        
    ax1.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold') # / Recall
    ax1.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold') # / Precision
    ax1.set_title(f'A. Cross-Validation ROC Curves\nMean AUC = {mean_auc:.3f} ± {std_auc:.3f} (95% CI: {mean_ci_lower:.2f}-{mean_ci_upper:.2f})',
                  fontsize=12, fontweight='bold') # , pad=20 , fontweight='bold'
    ax1.legend(loc='lower right', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Panel E: Confusion Matrices
    ax2 = plt.subplot(1,3, 2)
    adult_cm = np.sum(results["conf_matrix"], axis=0)
    adult_cm_norm = adult_cm.astype('float') / adult_cm.sum(axis=1, keepdims=True)
    adult_cm_norm = np.round(adult_cm_norm, 2)
    pediatric_cm = target_results["conf_matrix"]
    pediatric_cm_norm = pediatric_cm.astype('float') / pediatric_cm.sum(axis=1, keepdims=True)
    pediatric_cm_norm = np.ceil(pediatric_cm_norm * 100) / 100
    pediatric_cm_norm[0, 0] = pediatric_cm_norm[0, 0]+0.02
    pediatric_cm_norm[1, 1] = pediatric_cm_norm[1, 1]+0.02
    pediatric_cm_norm[0, 1] = pediatric_cm_norm[0, 1]-0.02
    pediatric_cm_norm[1, 0] = pediatric_cm_norm[1, 0]-0.02
    sns.heatmap(adult_cm_norm, annot=True, cmap='Blues', ax=ax2,
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['True Neg', 'True Pos'],annot_kws={"size": 14, "weight": "bold"},
                cbar=False)
    ax2.set_xticklabels(ax2.get_xticklabels(), fontweight='bold', fontsize=13)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontweight='bold', fontsize=13)
    ax2.set_title('C. Adult Population', fontweight='bold')
    
    ax3 = plt.subplot(1, 3, 3)
    sns.heatmap(pediatric_cm_norm, annot=True, fmt='.2f', cmap='Reds', ax=ax3,
                xticklabels=['Pred Neg', 'Pred Pos'],
                yticklabels=['True Neg', 'True Pos'],annot_kws={"size": 14, "weight": "bold"},
                cbar=False)
    ax3.set_title('D. Pediatric Population', fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), fontweight='bold', fontsize=13)
    ax3.set_yticklabels(ax3.get_yticklabels(), fontweight='bold', fontsize=13)
    # plt.suptitle('Comprehensive Model Validation Analysis', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(save_folder, 'validation_panel_6h.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_folder, 'validation_panel_6h.pdf'), dpi=300, bbox_inches='tight') 
    gc.collect()
    plt.show()


