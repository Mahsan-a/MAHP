
"""
Data preprocessing pipeline for DEXI dataset preparation.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import dill
from data_preprocessing import prepare_datasets
from feature_engineering import get_cgm_prediction_stats

def load_dexi_datasets():
    """Load all DEXI dataset components."""
    DEXI_folder = '/home/ma98/Datasets/Compressed_DEXI_Data/'
    DEXIP_folder = '/home/ma98/Datasets/Compressed_DEXIP_Data/'

    pediatric_datasets = {
        'cgm': pd.read_parquet(os.path.join(DEXIP_folder, 'T1DEXIP_Dataset.csv')),
        'basal': pd.read_parquet(os.path.join(DEXIP_folder, 'T1DEXIP_Insulin_Dataset.csv')),
        'carbs': pd.read_parquet(os.path.join(DEXIP_folder, 'T1DEXIP_Reqcue_Carbs_Dataset.csv')),
        'famlpm': pd.read_parquet(os.path.join(DEXIP_folder, 'P_FAMLPM_Dataset.csv')),
        'sleep': pd.read_parquet(os.path.join(DEXIP_folder, 'T1DEXIP_sleep_Dataset.csv')),
        'vitals': pd.read_csv(os.path.join(DEXIP_folder, 'T1DEXIP_VS_Dataset.csv.gz'))
    }
    
    # Process datetime columns
    pediatric_datasets['cgm']['LBDTC'] = pd.to_datetime(pediatric_datasets['cgm']['LBDTC'])
    pediatric_datasets['basal']['FADTC'] = pd.to_datetime(pediatric_datasets['basal']['FADTC'])
    pediatric_datasets['carbs']['MLDTC'] = pd.to_datetime(pediatric_datasets['carbs']['MLDTC'])
    pediatric_datasets['famlpm']['FADTC'] = pd.to_datetime(pediatric_datasets['famlpm']['FADTC'])
    pediatric_datasets['vitals']['VSDTC'] = pd.to_datetime(pediatric_datasets['vitals']['VSDTC'])

    adult_datasets = {
        'cgm': pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_Dataset.csv')),
        'basal': pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_Insulin_Dataset.csv')),
        'carbs': pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_Reqcue_Carbs_Dataset.csv')),
        'famlpm': pd.read_parquet(os.path.join(DEXI_folder, 'FAMLPM_Dataset.csv')),
        'sleep': pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_sleep_Dataset.csv')),
        'vitals': pd.read_csv(os.path.join(DEXI_folder, 'T1DEXI_VS_Dataset.csv.gz'))
    }
    
    # Process datetime columns
    adult_datasets['cgm']['LBDTC'] = pd.to_datetime(adult_datasets['cgm']['LBDTC'])
    adult_datasets['basal']['FADTC'] = pd.to_datetime(adult_datasets['basal']['FADTC'])
    adult_datasets['carbs']['MLDTC'] = pd.to_datetime(adult_datasets['carbs']['MLDTC'])
    adult_datasets['famlpm']['FADTC'] = pd.to_datetime(adult_datasets['famlpm']['FADTC'])
    adult_datasets['vitals']['VSDTC'] = pd.to_datetime(adult_datasets['vitals']['VSDTC'])
    
    return adult_datasets, pediatric_datasets

def main():
    """Main preprocessing pipeline."""
    print("Loading DEXI datasets...")
    adult_datasets, pediatric_datasets = load_dexi_datasets()
    
    # Find shared participants
    adult_shared_participants = sorted(
        set(adult_datasets['cgm']['USUBJID']) & 
        set(adult_datasets['basal']['USUBJID']) &
        set(adult_datasets['famlpm']['USUBJID']) & 
        set(adult_datasets['vitals']['USUBJID'])
    )
    pediatric_shared_participants = sorted(
        set(pediatric_datasets['cgm']['USUBJID']) & 
        set(pediatric_datasets['basal']['USUBJID']) &
        set(pediatric_datasets['famlpm']['USUBJID']) & 
        set(pediatric_datasets['vitals']['USUBJID'])
    )
    # data processing loop for each dataset
    
if __name__ == "__main__":
    main()
    

