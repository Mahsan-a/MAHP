
"""
Feature engineering and statistical calculations for glucose data.
"""

def resample_avg(data, times, interval='5T'):
    df_data = pd.DataFrame({'datetime': times, 'value': data})
    df_data.set_index('datetime', inplace=True)
    data_resampled = df_data.copy()
    data_resampled.index = data_resampled.index.floor(interval)
    data_resampled = data_resampled.groupby(data_resampled.index).mean()
    data_resampled = data_resampled.resample(interval).asfreq()
    return data_resampled

def smooth_signal(df, window_size=17, treshold=10, order=None):
    data = df['value']

    if order is not None:
        # Temporarily fill NaNs with interpolation to avoid issues with savgol_filter
        data_filled = data.interpolate(method='linear')
        # Apply savgol_filter only to non-NaN values
        smoothed_filled = savgol_filter(data_filled.fillna(0), window_length=window_size-8, polyorder=order)
        # Reintroduce NaNs where they were originally
        smoothed_values = pd.Series(smoothed_filled, index=data.index)
        smoothed_values[np.isnan(data)] = np.nan
    else:
        smoothed_values = data
        
    half_window = window_size // 2
    # Generate a Gaussian (normal) distribution for weights
    # The mean is at the center of the window, and the standard deviation determines the spread
    weights = norm.pdf(np.arange(-half_window, half_window + 1), 0, 2)
    # Normalize weights so that they sum to 1
    weights /= weights.sum()
    data_smooth = pd.Series(smoothed_values, index=df.index)
    rolling_avg = data_smooth.rolling(window=window_size, center=True).apply(lambda x: np.dot(x, weights), raw=True)

    fluctuations = data - rolling_avg
    smoothed_values = rolling_avg + fluctuations.where(fluctuations.abs() > treshold, 0)
    
    weights = norm.pdf(np.arange(-3, 3 + 1), 0, 2)
    weights /= weights.sum()
    rolling_avg = smoothed_values.rolling(window=7, center=True).apply(lambda x: np.dot(x, weights), raw=True)
    fluctuations = smoothed_values - rolling_avg
    smoothed_values = rolling_avg + fluctuations.where(fluctuations.abs() > treshold, 0)

    data_filled = smoothed_values.interpolate(method='linear')
    # Apply savgol_filter only to non-NaN values
    smoothed_filled = savgol_filter(data_filled.fillna(0), window_length=window_size-10, polyorder=order+1)
    # Reintroduce NaNs where they were originally
    smoothed_values = pd.Series(smoothed_filled, index=data.index)
    smoothed_values[np.isnan(data)] = np.nan
        
    smoothed_df = pd.DataFrame({'datetime': df.index, 'value': smoothed_values})
    smoothed_df.set_index('datetime', inplace=True)
    return smoothed_df

def remove_nan_rows(X_daily, X_individual, y_label):
    # Find rows with NaNs in either feature vector
    nan_mask_daily = np.any(np.isnan(X_daily), axis=(1, 2))
    nan_mask_individual = np.any(np.isnan(X_individual), axis=(1, 2))
    
    nan_mask_combined = nan_mask_daily | nan_mask_individual    
    # Keep only rows without NaNs
    X_daily_clean = X_daily[~nan_mask_combined]
    X_individual_clean = X_individual[~nan_mask_combined]
    y_label_clean = y_label[~nan_mask_combined]
    return X_daily_clean, X_individual_clean, y_label_clean

# Normalize the individual features
def standard_scale(data):
    scaler = StandardScaler()
    if np.isinf(data).any():
        print("There are infinite values in the data.")
        
        # Replace infinite values with NaN (or a large number)
        data[np.isinf(data)] = np.nan
    return scaler.fit_transform(data.reshape(-1, data.shape[-1]))


def check_nans(arr, name):
    print(f"{name}: {np.isnan(arr).sum()}  NaNs with shape: {arr.shape}")


    

