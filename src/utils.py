
"""
General utility functions.
"""

import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy import stats
from scipy.stats import spearmanr
from scipy.signal import find_peaks, detrend
import pywt
from numpy import arange, array, linspace, loadtxt, log2, logspace, mean, polyfit
from numpy import zeros, pi, sin, cos, arctan2, sqrt, real, imag, conj, tile
from numpy import round, interp, diff, unique, where
from pandas import DataFrame, date_range
import matplotlib.dates as mdates
from matplotlib import pyplot
from scipy.stats import pearsonr, mannwhitneyu, kruskal, norm, wasserstein_distance
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,f1_score, recall_score, precision_score,roc_auc_score
from sklearn.metrics import roc_curve, auc,accuracy_score,classification_report,confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import keras.backend as K
from keras.utils import Sequence

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from collections import defaultdict, Counter

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras import layers
# from tensorflow.keras.models import Model, load_model
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D,Concatenate,Input,Dropout, Conv2D, MaxPooling2D, Flatten,Multiply,Attention,Dense,concatenate,Masking,BatchNormalization, Reshape
from keras.models import Sequential
from keras.optimizers import Adam
from keras.applications import DenseNet121
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, History
history = History()

import pickle
import joblib
import dill
import json
from collections import defaultdict
from scipy.signal import savgol_filter
import gc
import time

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import sys
sys.stderr = open(os.devnull, 'w')


def create_directories(base_path):
    """Create necessary directories for results."""
    validation_folder = os.path.join(base_path, 'validation_results')
    os.makedirs(validation_folder, exist_ok=True)
    return validation_folder

def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except:
        pass

def save_results(results, target_results, save_path):
    """Save analysis results."""
    np.save(os.path.join(save_path, 'source_results.npy'), results)
    np.save(os.path.join(save_path, 'target_results.npy'), target_results)

