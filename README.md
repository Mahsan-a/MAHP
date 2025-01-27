# EPHIC Extended Prediction of Hypoglycemia by Integrating Continuous Wavelet Transform
EPHIC is a predictive framework that uses advanced signal processing techniques, and deep learning to leverage
multimodal data commonly collected as part of T1D self-management for extended hypoglycemia prediction.
The full dataframe is created from the seperate files in the T1DEXI dataset using all the physiological and behavioural data (CGM readings, basal insulin doses, meal timings with carbohy-
drate estimates, heart rate signals), and personal characteristics (demographics, diabetes history) to predict hypoglycemia.
Following all the preprocessing steps, the Multimodal CNN is implemented for the nocturnal hypoglycemia prediction in the "MultiCNN_Nocturnal.ipynb" and for other classification horizons in the "DA_Extended_Hypo.ipynb" files.
