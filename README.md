This repository contains the Jupyter Notebooks used for the analysis and result generation of our recent study on # MAHP Multi-modal Approach for Hypoglycemia Prediction 

MAHP is a predictive framework that uses advanced signal processing techniques, and deep learning to leverage multimodal data commonly collected as part of T1D self-management for extended hypoglycemia prediction.
The internal training, validation and testing dataframe is created from the seperate files in the T1DEXI dataset using all the physiological and behavioural data (CGM readings, basal insulin doses, meal timings with carbohydrate estimates, heart rate signals), and personal characteristics (demographics, diabetes history) to predict hypoglycemia. The external evaluation dataset dataset is created from the seperate files in the T1DEXIP dataset from a pediatric T1D dataset including the physiological and behavioural data, and personal characteristics.

The code here directly reproduces the figures and metrics reported in the manuscript. These notebooks focus on model training, evaluation, and result visualization for both adult test set and external evaluation on pediatric cohort.

Due to restrictions on sensitive data, the full dataset and preprocessing scripts are not fully shared here. If you're a researcher or student interested in extending this work, please contact me directly.
📧 Email:ma98@rice.edu

If you build upon this work, please cite.
