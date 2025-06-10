
"""
Complete training pipeline for 24-hour hypoglycemia prediction throughout the whole next day.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import DEFAULT_CONFIG
from training_utils import cross_validation_holdout_analysis
from validation_analysis import *
from visualization import *
from utils import *

def main():
    
    base_path = "results/complete_analysis_Hypo_Day"
    validation_folder = create_directories(base_path)
    
    results, target_results = cross_validation_holdout_analysis(
        label_name="hypo_day",
        num_splits=8, 
        num_epochs=40,
        batch_s=32, 
        num_patience=12, 
        monitored='val_AUC'
    )
    
    optimal = create_threshold_analysis_plot(
        results["fold_y_true"], 
        results["fold_y_pred_proba"],
        validation_folder
    )
    
    ROC_full_figure_panel(results, target_results, validation_folder)
    ROC_figure_panel(results, target_results, validation_folder)
    
    threshold_figure_panel(
        results["fold_y_true"], 
        results["fold_y_pred_proba"], 
        np.mean(results["recall_scores"]),
        np.mean(results["specificity_scores"]),
        results["calibration_data"],
        target_results["calibration_data"],
        validation_folder
    )
    
    create_validation_report(results, target_results, validation_folder)
    create_calibration_plots_from_results(results["calibration_data"], validation_folder)
    create_calibration_panel(results, target_results, validation_folder)
    
    save_results(results, target_results, validation_folder)
    
    print(f"\nAll validation analyses completed!")
    print(f"Results saved in: {validation_folder}")
    
    cleanup_memory()

if __name__ == "__main__":
    main()
    

