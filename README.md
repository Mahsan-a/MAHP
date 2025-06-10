This repository contains the implementation of a multi-modal convolutional neural network designed for early prediction of hypoglycemic events in Type 1 diabetes patients. The model integrates time-frequency representations of glucose data with clinical features to provide robust predictive capabilities across different patient populations.


MAHP is a predictive framework that uses advanced signal processing techniques, and deep learning to leverage multimodal data commonly collected as part of T1D self-management for extended hypoglycemia prediction.
The internal training, validation and testing dataframe is created from the seperate files in the T1DEXI dataset using all the physiological and behavioural data (CGM readings, basal insulin doses, meal timings with carbohydrate estimates, heart rate signals), and personal characteristics (demographics, diabetes history) to predict hypoglycemia. The external evaluation dataset dataset is created from the seperate files in the T1DEXIP dataset from a pediatric T1D dataset including the physiological and behavioural data, and personal characteristics.

## Key Features

- **Multi-modal Architecture**: Combines CNN for temporal pattern recognition with dense networks for clinical features
- **Cross-Population Validation**: Validated on both adult and pediatric populations
- **Multiple Prediction Horizons**: Supports 6-hour, 8-hour, 12-hour, and 24-hour prediction windows
- **Comprehensive Validation**: Includes ROC analysis, calibration assessment, and feature importance evaluation
- **Clinical Focus**: Optimized for high sensitivity to minimize missed hypoglycemic events

The code here directly reproduces the figures and metrics reported in the manuscript. 
Due to restrictions on sensitive data, the full dataset and preprocessing scripts are not fully shared here. If you're a researcher or student interested in extending this work, please contact me directly.
📧 Email:ma98@rice.edu

If you build upon this work, please cite.

## Installation

```bash
git clone https://github.com/Mahsan-a/MAHP
cd MAHP
pip install -r requirements.txt
