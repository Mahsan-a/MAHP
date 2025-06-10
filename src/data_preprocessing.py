
"""
Feature engineering and statistical calculations for glucose data.
"""

def get_cgm_prediction_stats(current_day_cgm):
    general_stats = calculate_stats(current_day_cgm.values[:,0])
    general_stats = extract_prediction_stats(general_stats, 'cgm')
    masks = {
        'night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 6),
        'early_night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 3),
        'late_night': (current_day_cgm.index.hour >= 3) & (current_day_cgm.index.hour < 6),
        'long_night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 8),
        'night_morning': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 12),
        'morning': (current_day_cgm.index.hour >= 6) & (current_day_cgm.index.hour < 12),
        'afternoon': (current_day_cgm.index.hour >= 12) & (current_day_cgm.index.hour < 18),
        'evening': (current_day_cgm.index.hour >= 18) & (current_day_cgm.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(current_day_cgm, mask)
        period_stats = extract_prediction_stats(period_stats, f'cgm_{period}')
        time_range_stats.update(period_stats)
    
    time_in_range_stats = {
        'time_in_range_70_140_day': calculate_time_in_range(current_day_cgm, 70, 140),
        'time_in_range_70_180_day': calculate_time_in_range(current_day_cgm, 70, 180),
        'time_below_range_70_day': calculate_time_in_range(current_day_cgm, 0, 70),
        'time_above_range_140_day': calculate_time_in_range(current_day_cgm, 140, np.inf),
        'time_above_range_180_day': calculate_time_in_range(current_day_cgm, 180, np.inf)
    }
    for period, mask in masks.items():
        time_in_range_stats[f'time_in_range_70_140_{period}'] = calculate_time_in_range(current_day_cgm, 70, 140, mask)
        time_in_range_stats[f'time_in_range_70_180_{period}'] = calculate_time_in_range(current_day_cgm, 70, 180, mask)
        time_in_range_stats[f'time_below_range_70_{period}'] = calculate_time_in_range(current_day_cgm, 0, 70, mask)
        time_in_range_stats[f'time_above_range_140_{period}'] = calculate_time_in_range(current_day_cgm, 140, np.inf, mask)
        time_in_range_stats[f'time_above_range_180_{period}'] = calculate_time_in_range(current_day_cgm, 180, np.inf, mask)
    
    cgm_prediction_stats = pd.DataFrame({**general_stats, **time_range_stats, **time_in_range_stats})
    return cgm_prediction_stats.astype(np.float16)
    
def get_daily_cgm_stats(current_day_cgm):
    general_stats = calculate_stats(current_day_cgm.values[:,0])
    general_stats = extract_stats(general_stats, 'daily_cgm')
    mean_crossings = calculate_crossings(current_day_cgm.values[:,0])
    general_stats['daily_cgm_mean_crossings'] = mean_crossings

    # Add new features
    last_cgm_value = current_day_cgm.values[-1, 0]
    if last_cgm_value == np.nan:
        last_cgm_value = current_day_cgm.values[-2, 0]
    if last_cgm_value == np.nan:
        last_cgm_value = current_day_cgm.values[-3, 0]
    if last_cgm_value == np.nan:
        last_cgm_value = current_day_cgm.values[-4, 0]
    general_stats['last_cgm_value'] = last_cgm_value #11:55
    general_stats['last_cgm_sq'] = np.sqrt(last_cgm_value)
    
    diff_10_min = last_cgm_value - current_day_cgm.values[-3, 0]
    general_stats['diff_10_min'] = diff_10_min

    diff_20_min = last_cgm_value - current_day_cgm.values[-5, 0]
    general_stats['diff_20_min'] = diff_20_min

    diff_30_min = last_cgm_value - current_day_cgm.values[-7, 0]
    general_stats['diff_30_min'] = diff_30_min

    diff_50_min = last_cgm_value - current_day_cgm.values[-11, 0]
    general_stats['diff_50_min'] = diff_50_min
    
    # Rate of change (slope) in CGM in the last hours
    slope_last_half_hour = (last_cgm_value - current_day_cgm.values[-7, 0]) / last_cgm_value
    general_stats['slope_last_half_hour'] = slope_last_half_hour
    slope_last_hour = (last_cgm_value - current_day_cgm.values[-13, 0]) / last_cgm_value
    general_stats['slope_last_hour'] = slope_last_hour
    slope_last_2_hour = (last_cgm_value - current_day_cgm.values[-25, 0]) / last_cgm_value
    general_stats['slope_last_2_hour'] = slope_last_2_hour
    
    # SD of CGM in the last hours
    std_last_2_hours = np.std(current_day_cgm.values[-25:, 0])
    general_stats['std_last_2_hours'] = std_last_2_hours
    std_last_hour = np.std(current_day_cgm.values[-13:, 0])
    general_stats['std_last_hour'] = std_last_hour
    
    # Sum of all increments in adjacent CGM observations in the last two hours
    increments_sum_last_2_hours = np.sum(np.diff(current_day_cgm.values[-25:, 0])[np.diff(current_day_cgm.values[-25:, 0]) > 0])
    general_stats['increments_sum_last_2_hours'] = increments_sum_last_2_hours

    # Sum of all decrements in adjacent CGM observations in the last two hours
    decrements_sum_last_2_hours = np.sum(np.diff(current_day_cgm.values[-25:, 0])[np.diff(current_day_cgm.values[-25:, 0]) < 0])
    general_stats['decrements_sum_last_2_hours'] = decrements_sum_last_2_hours

    # Maximum increase in adjacent CGM observations in the last two hours
    max_increase_last_2_hours = np.max(np.diff(current_day_cgm.values[-25:, 0]))
    general_stats['max_increase_last_2_hours'] = max_increase_last_2_hours

    # Maximum decrease in adjacent CGM observations in the last two hours
    max_decrease_last_2_hours = np.min(np.diff(current_day_cgm.values[-25:, 0]))
    general_stats['max_decrease_last_2_hours'] = max_decrease_last_2_hours

    masks = {
        'night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 6),
        'morning': (current_day_cgm.index.hour >= 6) & (current_day_cgm.index.hour < 12),
        'afternoon': (current_day_cgm.index.hour >= 12) & (current_day_cgm.index.hour < 18),
        'evening': (current_day_cgm.index.hour >= 18) & (current_day_cgm.index.hour < 24),
        'late_evening': (current_day_cgm.index.hour >= 21) & (current_day_cgm.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(current_day_cgm, mask)
        period_stats = extract_stats(period_stats, f'daily_cgm_{period}')
        time_range_stats.update(period_stats)
    
    time_in_range_stats = {
        'daily_time_in_range_70_140_day': calculate_time_in_range(current_day_cgm, 70, 140),
        'daily_time_in_range_70_180_day': calculate_time_in_range(current_day_cgm, 70, 180),
        'daily_time_below_range_70_day': calculate_time_in_range(current_day_cgm, 0, 70),
        'daily_time_above_range_140_day': calculate_time_in_range(current_day_cgm, 140, np.inf),
        'daily_time_above_range_180_day': calculate_time_in_range(current_day_cgm, 180, np.inf)
    }
    for period, mask in masks.items():
        time_in_range_stats[f'daily_time_in_range_70_140_{period}'] = calculate_time_in_range(current_day_cgm, 70, 140, mask)
        time_in_range_stats[f'daily_time_in_range_70_180_{period}'] = calculate_time_in_range(current_day_cgm, 70, 180, mask)
        time_in_range_stats[f'daily_time_below_range_70_{period}'] = calculate_time_in_range(current_day_cgm, 0, 70, mask)
        time_in_range_stats[f'daily_time_above_range_140_{period}'] = calculate_time_in_range(current_day_cgm, 140, np.inf, mask)
        time_in_range_stats[f'daily_time_above_range_180_{period}'] = calculate_time_in_range(current_day_cgm, 180, np.inf, mask)
    
    daily_cgm_stats = pd.DataFrame({**general_stats, **time_range_stats, **time_in_range_stats})
    return daily_cgm_stats

def get_daily_cgm_stats(current_day_cgm):
    general_stats = calculate_stats(current_day_cgm.values[:,0])
    general_stats = extract_stats(general_stats, 'daily_cgm')
    mean_crossings = calculate_crossings(current_day_cgm.values[:,0])
    general_stats['daily_cgm_mean_crossings'] = mean_crossings

    # Add new features
    last_cgm_value = current_day_cgm.values[-1, 0]
    general_stats['last_cgm_value'] = last_cgm_value #11:55
    general_stats['last_cgm_sq'] = np.sqrt(last_cgm_value)
    
    general_stats['diff_10_min'] = last_cgm_value - current_day_cgm.values[-3, 0] #11:45
    general_stats['diff_20_min'] = last_cgm_value - current_day_cgm.values[-5, 0] #11:35
    general_stats['diff_30_min'] = last_cgm_value - current_day_cgm.values[-7, 0] #11:25
    general_stats['diff_40_min'] = last_cgm_value - current_day_cgm.values[-9, 0] #11:15
    general_stats['diff_50_min'] = last_cgm_value - current_day_cgm.values[-11, 0] #11:05
    
    # Rate of change (slope) in CGM in the last hours
    general_stats['slope_last_half_hour'] = (last_cgm_value - current_day_cgm.values[-6, 0]) / last_cgm_value #11:30
    general_stats['slope_last_hour'] = (last_cgm_value - current_day_cgm.values[-13, 0]) / last_cgm_value #10:55
    general_stats['slope_last_hour_half'] = (last_cgm_value - current_day_cgm.values[-19, 0]) / last_cgm_value #10:25
    general_stats['slope_last_2_hour'] = (last_cgm_value - current_day_cgm.values[-25, 0]) / last_cgm_value #9:55
    
    # SD of CGM in the last hours
    general_stats['std_last_2_hours'] = np.std(current_day_cgm.values[-25:, 0]) #9:55
    general_stats['std_last_hour'] = np.std(current_day_cgm.values[-13:, 0]) #10:55
    
    # Sum of all increments in adjacent CGM observations in the last two hours
    increments_sum_last_2_hours = np.sum(np.diff(current_day_cgm.values[-25:, 0])[np.diff(current_day_cgm.values[-25:, 0]) > 0])
    general_stats['increments_sum_last_2_hours'] = increments_sum_last_2_hours

    # Sum of all decrements in adjacent CGM observations in the last two hours
    decrements_sum_last_2_hours = np.sum(np.diff(current_day_cgm.values[-25:, 0])[np.diff(current_day_cgm.values[-25:, 0]) < 0])
    general_stats['decrements_sum_last_2_hours'] = decrements_sum_last_2_hours

    # Maximum increase in adjacent CGM observations in the last two hours
    max_increase_last_2_hours = np.max(np.diff(current_day_cgm.values[-25:, 0]))
    general_stats['max_increase_last_2_hours'] = max_increase_last_2_hours

    # Maximum decrease in adjacent CGM observations in the last two hours
    max_decrease_last_2_hours = np.min(np.diff(current_day_cgm.values[-25:, 0]))
    general_stats['max_decrease_last_2_hours'] = max_decrease_last_2_hours
    masks = {
        'night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 6),
        'morning': (current_day_cgm.index.hour >= 6) & (current_day_cgm.index.hour < 12),
        'afternoon': (current_day_cgm.index.hour >= 12) & (current_day_cgm.index.hour < 18),
        'evening': (current_day_cgm.index.hour >= 18) & (current_day_cgm.index.hour < 24),
        'late_evening': (current_day_cgm.index.hour >= 21) & (current_day_cgm.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(current_day_cgm, mask)
        period_stats = extract_stats(period_stats, f'daily_cgm_{period}')
        time_range_stats.update(period_stats)

    time_in_range_stats = {
        'daily_time_in_range_70_140_day': calculate_time_in_range(current_day_cgm, 70, 140),
        'daily_time_in_range_70_180_day': calculate_time_in_range(current_day_cgm, 70, 180),
        'daily_time_below_range_70_day': calculate_time_in_range(current_day_cgm, 0, 70),
        'daily_time_above_range_140_day': calculate_time_in_range(current_day_cgm, 140, np.inf),
        'daily_time_above_range_180_day': calculate_time_in_range(current_day_cgm, 180, np.inf)
    }
    for period, mask in masks.items():
        time_in_range_stats[f'daily_time_in_range_70_140_{period}'] = calculate_time_in_range(current_day_cgm, 70, 140, mask)
        time_in_range_stats[f'daily_time_in_range_70_180_{period}'] = calculate_time_in_range(current_day_cgm, 70, 180, mask)
        time_in_range_stats[f'daily_time_below_range_70_{period}'] = calculate_time_in_range(current_day_cgm, 0, 70, mask)
        time_in_range_stats[f'daily_time_above_range_140_{period}'] = calculate_time_in_range(current_day_cgm, 140, np.inf, mask)
        time_in_range_stats[f'daily_time_above_range_180_{period}'] = calculate_time_in_range(current_day_cgm, 180, np.inf, mask)
    
    daily_cgm_stats = pd.DataFrame({**general_stats, **time_range_stats, **time_in_range_stats})
    return daily_cgm_stats.astype(np.float16)
    
def get_total_cgm_stats(current_day_cgm):
    general_stats = calculate_stats(current_day_cgm.values[:,0])
    general_stats = extract_stats(general_stats, 'total_cgm')
    mean_crossings = calculate_crossings(current_day_cgm.values[:,0])
    general_stats['cgm_mean_crossings'] = mean_crossings
    masks = {
        'night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 6),
        'early_night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 3),
        'late_night': (current_day_cgm.index.hour >= 3) & (current_day_cgm.index.hour < 6),
        'morning': (current_day_cgm.index.hour >= 6) & (current_day_cgm.index.hour < 12),
        'afternoon': (current_day_cgm.index.hour >= 12) & (current_day_cgm.index.hour < 18),
        'evening': (current_day_cgm.index.hour >= 18) & (current_day_cgm.index.hour < 24),
        'late_evening': (current_day_cgm.index.hour >= 21) & (current_day_cgm.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(current_day_cgm, mask)
        period_stats = extract_stats(period_stats, f'total_cgm_{period}')
        time_range_stats.update(period_stats)
    
    time_in_range_stats = {
        'total_time_in_range_70_140_day': calculate_time_in_range(current_day_cgm, 70, 140),
        'total_time_in_range_70_170_day': calculate_time_in_range(current_day_cgm, 70, 180),
        'total_time_below_range_70_day': calculate_time_in_range(current_day_cgm, 0, 70),
        'total_time_above_range_140_day': calculate_time_in_range(current_day_cgm, 140, np.inf),
        'total_time_above_range_180_day': calculate_time_in_range(current_day_cgm, 180, np.inf)
    }
    for period, mask in masks.items():
        time_in_range_stats[f'total_time_in_range_70_140_{period}'] = calculate_time_in_range(current_day_cgm, 70, 140, mask)
        time_in_range_stats[f'total_time_in_range_70_180_{period}'] = calculate_time_in_range(current_day_cgm, 70, 180, mask)
        time_in_range_stats[f'total_time_below_range_70_{period}'] = calculate_time_in_range(current_day_cgm, 0, 70, mask)
        time_in_range_stats[f'total_time_above_range_140_{period}'] = calculate_time_in_range(current_day_cgm, 140, np.inf, mask)
        time_in_range_stats[f'total_time_above_range_180_{period}'] = calculate_time_in_range(current_day_cgm, 180, np.inf, mask)
    
    total_cgm_stats = pd.DataFrame({**general_stats, **time_range_stats, **time_in_range_stats})
    return total_cgm_stats.astype(np.float16)
    
def daily_cgm_labels(current_day_cgm):
    masks = {
        'night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 6),
        'early_night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 3),
        'late_night': (current_day_cgm.index.hour >= 3) & (current_day_cgm.index.hour < 6),
        'long_night': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 8),
        'night_morning': (current_day_cgm.index.hour >= 0) & (current_day_cgm.index.hour < 12),
        'morning': (current_day_cgm.index.hour >= 6) & (current_day_cgm.index.hour < 12),
        'afternoon': (current_day_cgm.index.hour >= 12) & (current_day_cgm.index.hour < 18),
        'evening': (current_day_cgm.index.hour >= 18) & (current_day_cgm.index.hour < 24)
    }
    labels = {
        'hyper_day': return_label(current_day_cgm.values[:,0], 180),
        'hypo_day': return_label(current_day_cgm.values[:,0], 70)
    }
    for period, mask in masks.items():
        labels[f'hyper_{period}'] = return_label(current_day_cgm.values[:,0], 180, mask)
        labels[f'hypo_{period}'] = return_label(current_day_cgm.values[:,0], 70, mask)
    
    daily_cgm_labels = pd.DataFrame({**labels}, index=[0])
    return daily_cgm_labels
    
# --- Calculate Time in Range for Different Ranges and Periods ---
def calculate_time_in_range(glucose_data, range_lower, range_upper, period_mask=None):

    if period_mask is not None:
        glucose_data = glucose_data[period_mask]
    in_range = (glucose_data <= range_upper) & (glucose_data >= range_lower)
    return 100 * in_range.sum() / len(glucose_data) if len(glucose_data) > 0 else 0
    
def return_label(glucose_data, range, period_mask=None):
    if period_mask is not None:
        glucose_data = glucose_data[period_mask]
    if range==180:
        return np.sum(glucose_data>range)>2
    if range==70:
        return np.sum(glucose_data<=range)>1
        
def calculate_sleep_statistics(daily_sleep_data):
    participant_total_sleep = participant_sleep_data[participant_sleep_data['NVTEST'] == "Total Sleep Time"]
    participant_deep_sleep = participant_sleep_data[participant_sleep_data['NVTEST'] == "Deep NREM Duration"]
    participant_light_sleep = participant_sleep_data[participant_sleep_data['NVTEST'] == "Light NREM Duration"]
    participant_NREM_sleep = participant_sleep_data[participant_sleep_data['NVTEST'] == "NREM Duration"]
    participant_REM_sleep = participant_sleep_data[participant_sleep_data['NVTEST'] == "REM Duration"]
    participant_efficiency = participant_sleep_data[participant_sleep_data['NVTEST'] == "Sleep Efficiency"]
    participant_awakenings = participant_sleep_data[participant_sleep_data['NVTEST'] == "Number of Awakenings"]
    participant_latency = participant_sleep_data[participant_sleep_data['NVTEST'] == "Sleep Onset Latency"]

    avg_total_sleep = (participant_total_sleep['NVORRES']).mean()
    std_total_sleep = (participant_total_sleep['NVORRES']).std()
    avg_deep_sleep = (participant_deep_sleep['NVORRES']).mean()
    std_deep_sleep = (participant_deep_sleep['NVORRES']).std()
    avg_light_sleep = (participant_light_sleep['NVORRES']).mean()
    std_light_sleep = (participant_light_sleep['NVORRES']).std()
    avg_NREM_sleep = (participant_NREM_sleep['NVORRES']).mean()
    std_NREM_sleep = (participant_NREM_sleep['NVORRES']).std()
    avg_REM_sleep = (participant_REM_sleep['NVORRES']).mean()
    std_REM_sleep = (participant_REM_sleep['NVORRES']).std()
    avg_awakenings = (participant_awakenings['NVORRES']*100000.).mean()
    std_awakenings = (participant_awakenings['NVORRES']*100000. ).std()
    avg_latency = (participant_latency['NVORRES']).mean()
    std_latency = (participant_latency['NVORRES']).std()
    avg_efficiency = (participant_efficiency['NVORRES']*100000.).mean()
    std_efficiency = (participant_efficiency['NVORRES']*100000.).std()
    # Calculate avg deviation from midnight
    bedtime_deviations_midnight = []
    wakeup_deviations_midnight = []
    for dt in participant_total_sleep['NVDTC']:
        sleep_time = 60*dt.hour + (dt.minute)
        
        if sleep_time >720:
            sleep_time -= 1440
        bedtime_deviations_midnight.append(sleep_time)
    
    for dt in participant_total_sleep['NVENDTC']:
        wakeup_time = 60*dt.hour + (dt.minute)
        wakeup_deviations_midnight.append(wakeup_time)
  
    average_bedtime_midnight = np.mean(bedtime_deviations_midnight)
    average_wakeup_midnight = np.mean(wakeup_deviations_midnight)
    std_bedtime_midnight = np.std(bedtime_deviations_midnight)
    std_wakeup_midnight =  np.std(wakeup_deviations_midnight)
   #     # Convert average deviation back to time format
    average_bedtime = pd.to_datetime('00:00:00') + pd.to_timedelta(average_bedtime_midnight, unit='m')
    average_wakeup = pd.to_datetime('00:00:00') + pd.to_timedelta(average_wakeup_midnight, unit='m')
       
    # Calculate bedtime deviations from average & variance & Consistency Score (inverse of variance)
    participant_sleep_data['bedtime_from_avg'] = ((60*participant_sleep_data['NVDTC'].dt.hour + participant_sleep_data['NVDTC'].dt.minute) - average_bedtime_midnight)
    participant_sleep_data.loc[participant_sleep_data['bedtime_from_avg'] > 720, 'bedtime_from_avg'] -= 1440

    participant_sleep_data['wakeup_from_avg'] = ((60*participant_sleep_data['NVENDTC'].dt.hour + participant_sleep_data['NVENDTC'].dt.minute) - average_wakeup_midnight)
    bedtime_std = np.round(participant_sleep_data['bedtime_from_avg'].std(),3)
    wakeup_std = np.round(participant_sleep_data['wakeup_from_avg'].std(),3)
    bedtime_var = np.round(participant_sleep_data['bedtime_from_avg'].var(),3)
    wakeup_var = np.round(participant_sleep_data['wakeup_from_avg'].var(),3)
    sleep_stats = pd.DataFrame({
        'avg_bedtime_midnight': [average_bedtime_midnight.round(3)],
        'avg_bedtime': [average_bedtime],
        'bedtime_consistency_score': [(100. / bedtime_var).round(3)],
        'bedtime_std': [bedtime_std],
        'bedtime_var': [bedtime_var],
        'avg_wakeup_midnight': [average_wakeup_midnight.round(3)],
        'avg_wakeup': [average_wakeup],
        'wakeup_consistency_score': [(100. / wakeup_var).round(3)],
        'wakeup_std': [wakeup_std],
        'wakeup_var': [wakeup_var],
        'avg_deep_sleep': [round(avg_deep_sleep, 3)],
        'std_deep_sleep': [round(std_deep_sleep, 3)],
        'avg_total_sleep': [round(avg_total_sleep, 3)],
        'std_total_sleep': [round(std_total_sleep, 3)],
        'avg_light_sleep': [round(avg_light_sleep, 3)],
        'std_light_sleep': [round(std_light_sleep, 3)],
        'avg_NREM_sleep': [round(avg_NREM_sleep, 3)],
        'std_NREM_sleep': [round(std_NREM_sleep, 3)],
        'avg_REM_sleep': [round(avg_REM_sleep, 3)],
        'std_REM_sleep': [round(std_REM_sleep, 3)],
        'avg_awakenings': [round(avg_awakenings, 3)],
        'std_awakenings': [round(std_awakenings, 3)],
        'avg_latency': [round(avg_latency, 3)],
        'std_latency': [round(std_latency, 3)],
        'avg_efficiency': [round(avg_efficiency, 3)],
        'std_efficiency': [round(std_efficiency, 3)]
    })
    return participant_sleep_data, sleep_stats.astype(np.float16)
    
def calculate_stats(data, period_mask=None):
    if period_mask is not None:
        data = data[period_mask]
        data = data.values[:,0]
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return pd.DataFrame({
            'avg': [np.nan],
            'std': [np.nan],
            'min': [np.nan],
            'n5': [np.nan],
            'n25': [np.nan],
            'median': [np.nan],
            'n75': [np.nan],
            'n95': [np.nan],
            'max': [np.nan],
            'var': [np.nan],
            'std_score': [np.nan],
            'var_score': [np.nan],
            'consistency_score': [np.nan],
            'entropy': [np.nan]
        })
    max_val = round(np.max(data), 3)
    min_val = round(np.min(data), 3)
    avg_val = round(np.mean(data), 3)
    median = round(np.percentile(data, 50),3)
    
    if len(data) > 1:
        n5 = round(np.percentile(data, 5), 3)
        n25 = round(np.percentile(data, 25), 3)
        n75 = round(np.percentile(data, 75), 3)
        n95 = round(np.percentile(data, 95), 3)
        std_val = round(np.std(data), 3)
        var_val = round(np.var(data), 3)
        std_score_val = round(100 * np.std(data) / np.mean(data), 3)
        var_score_val = round(np.var(data) / np.mean(data), 3)
        consistency_score_val = round(100. / np.var(data), 3) if np.var(data) != 0 else np.nan
        entropy = calculate_entropy(data)
    else:
        n5 = np.nan
        n25 = np.nan
        n75 = np.nan
        n95 = np.nan
        std_val = np.nan
        var_val = np.nan
        std_score_val = np.nan
        var_score_val = np.nan
        consistency_score_val = np.nan
        entropy = np.nan
    data_stats = pd.DataFrame({
        'avg': [avg_val],
        'std': [std_val],
        'min': [min_val],
        'n5': [n5],
        'n25': [n25],
        'median': [median],
        'n75': [n75],
        'n95': [n95],
        'max': [max_val],
        'var': [var_val],
        'std_score': [std_score_val],
        'var_score': [var_score_val],
        'consistency_score': [consistency_score_val],
        'entropy': [entropy]
    })
    return data_stats.astype(np.float16)
    
# Helper function to extract values from the DataFrame
def extract_stats(stats_df, prefix):
    return {
        f'{prefix}_avg': float(stats_df['avg'].values[0]),
        f'{prefix}_std': float(stats_df['std'].values[0]),
        f'{prefix}_min': float(stats_df['min'].values[0]),
        f'{prefix}_n5': float(stats_df['n5'].values[0]),
        f'{prefix}_n25': float(stats_df['n25'].values[0]),
        f'{prefix}_median': float(stats_df['median'].values[0]),
        f'{prefix}_n75': float(stats_df['n75'].values[0]),
        f'{prefix}_n95': float(stats_df['n95'].values[0]),
        f'{prefix}_max': float(stats_df['max'].values[0]),
        f'{prefix}_var': float(stats_df['var'].values[0]),
        f'{prefix}_std_score': float(stats_df['std_score'].values[0]),
        f'{prefix}_var_score': float(stats_df['var_score'].values[0]),
        f'{prefix}_consistency_score': float(stats_df['consistency_score'].values[0]),
        f'{prefix}_entropy': float(stats_df['entropy'].values[0]),
    }
    
def extract_prediction_stats(stats_df, prefix):
    return {
        f'{prefix}_max': float(stats_df['max'].values[0]),
        f'{prefix}_median': float(stats_df['median'].values[0]),
        f'{prefix}_avg': float(stats_df['avg'].values[0]),
        f'{prefix}_std_score': float(stats_df['std'].values[0])
    }
    
def extract_reduced_stats(stats_df, prefix):
    return {
        f'{prefix}_median': float(stats_df['median'].values[0]),
        f'{prefix}_avg': float(stats_df['avg'].values[0]),
        f'{prefix}_entropy': float(stats_df['entropy'].values[0]),
        f'{prefix}_std_score': float(stats_df['std'].values[0]),
        f'{prefix}_var_score': float(stats_df['var'].values[0])
    }
    
def calculate_sum(carbs_data, period_mask=None):
    if period_mask is not None:
        carbs_data = carbs_data[period_mask]
    return np.sum(carbs_data['corrected_value']) if not carbs_data['corrected_value'].empty else 0
    
def corrected_daily_carbs_stats(daily_carb_data):
    overall_stats = calculate_stats(daily_carb_data['corrected_value'].values)
    overall_stats = extract_prediction_stats(overall_stats, 'carbs')
    
    if len(daily_carb_data) > 0:
        # Calculate carb on board at midnight considering only intake after 7 PM
        start_time = pd.Timestamp(daily_carb_data['FADTC'].dt.date.iloc[0]) + pd.Timedelta(hours=19)  # 7 PM
        midnight = pd.Timestamp(daily_carb_data['FADTC'].dt.date.iloc[0]) + pd.Timedelta(days=1)  # 12 AM next day
        carb_on_board = 0.0
        for _, row in daily_carb_data[daily_carb_data['FADTC'] >= start_time].iterrows():
            carb_amount = row['corrected_value']
            intake_time = row['FADTC']
            absorption_start_time = intake_time + pd.Timedelta(minutes=15)

            if midnight > absorption_start_time:
                absorbed_duration = min((midnight - absorption_start_time).total_seconds() / 60, carb_amount / 0.5)
                absorbed_carbs = absorbed_duration * 0.5
                remaining_carbs = max(carb_amount - absorbed_carbs, 0)
                carb_on_board += remaining_carbs
    else:
        carb_on_board = np.nan
    overall_stats['carb_on_board_midnight'] = carb_on_board
    masks = {
        'night': (daily_carb_data['FADTC'].dt.hour >= 0) & (daily_carb_data['FADTC'].dt.hour < 6),
        'morning': (daily_carb_data['FADTC'].dt.hour >= 6) & (daily_carb_data['FADTC'].dt.hour < 12),
        'afternoon': (daily_carb_data['FADTC'].dt.hour >= 12) & (daily_carb_data['FADTC'].dt.hour < 18),
        'evening': (daily_carb_data['FADTC'].dt.hour >= 18) & (daily_carb_data['FADTC'].dt.hour < 24),
        'late_evening': (daily_carb_data['FADTC'].dt.hour >= 21) & (daily_carb_data['FADTC'].dt.hour < 24)
    } 
    time_range_stats = {}
    if len(daily_carb_data) > 0:
        time_range_stats['carbs_day'] = calculate_sum(daily_carb_data)
    else:
        time_range_stats['carbs_day'] = np.nan
        
    for period, mask in masks.items():
        if len(daily_carb_data) > 0:
            time_range_stats[f'carbs_{period}'] = calculate_sum(daily_carb_data, mask)
        else:
            time_range_stats[f'carbs_{period}'] = np.nan

    carbs_stats = pd.DataFrame({**overall_stats, **time_range_stats}, index=[0])
    return carbs_stats.astype(np.float16)
    
def total_stats(data, times, label_data='bolus'):
    daily_data = pd.DataFrame({'datetime': times, 'value': data})
    daily_data.set_index('datetime', inplace=True)
    
    general_stats = calculate_stats(daily_data.values[:,0])
    general_stats = extract_stats(general_stats, label_data)
    masks = {
        'night': (daily_data.index.hour >= 0) & (daily_data.index.hour < 6),
        'morning': (daily_data.index.hour >= 6) & (daily_data.index.hour < 12),
        'afternoon': (daily_data.index.hour >= 12) & (daily_data.index.hour < 18),
        'evening': (daily_data.index.hour >= 18) & (daily_data.index.hour < 24),
        'late_evening': (daily_data.index.hour >= 22) & (daily_data.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(daily_data, mask)
        period_stats = extract_stats(period_stats, f'{label_data}_{period}')
        time_range_stats.update(period_stats)
    
    total_stats = pd.DataFrame({**general_stats, **time_range_stats}, index=[0])
    return total_stats.astype(np.float16)
    
def daily_stats_hr(data, times, label_data='hr'):
    daily_data = pd.DataFrame({'datetime': times, 'value': data})
    daily_data.set_index('datetime', inplace=True)
    
    general_stats = calculate_stats(daily_data.values[:,0])
    general_stats = extract_stats(general_stats, label_data)
    mean_crossings = calculate_crossings(daily_data.values[:,0])
    general_stats[f'{label_data}_mean_crossings'] = mean_crossings
    masks = {
        'night': (daily_data.index.hour >= 0) & (daily_data.index.hour < 6),
        'morning': (daily_data.index.hour >= 6) & (daily_data.index.hour < 12),
        'afternoon': (daily_data.index.hour >= 12) & (daily_data.index.hour < 18),
        'evening': (daily_data.index.hour >= 18) & (daily_data.index.hour < 24),
        'late_evening': (daily_data.index.hour >= 21) & (daily_data.index.hour < 24),
        'before_sleep': (daily_data.index.hour >= 23) & (daily_data.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(daily_data, mask)
        period_stats = extract_stats(period_stats, f'{label_data}_{period}')
        time_range_stats.update(period_stats)
    
    daily_stats = pd.DataFrame({**general_stats, **time_range_stats}, index=[0])
    return daily_stats.astype(np.float16)
    
def daily_stats(data, times, label_data='basal'):
    daily_data = pd.DataFrame({'datetime': times, 'value': data})
    daily_data.set_index('datetime', inplace=True)
    
    general_stats = calculate_stats(daily_data.values[:,0])
    general_stats = extract_stats(general_stats, label_data)
    mean_crossings = calculate_crossings(daily_data.values[:,0])
    general_stats[f'{label_data}_mean_crossings'] = mean_crossings
    masks = {
        'night': (daily_data.index.hour >= 0) & (daily_data.index.hour < 6),
        'morning': (daily_data.index.hour >= 6) & (daily_data.index.hour < 12),
        'afternoon': (daily_data.index.hour >= 12) & (daily_data.index.hour < 18),
        'evening': (daily_data.index.hour >= 18) & (daily_data.index.hour < 24),
        'late_evening': (daily_data.index.hour >= 22) & (daily_data.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(daily_data, mask)
        period_stats = extract_stats(period_stats, f'{label_data}_{period}')
        time_range_stats.update(period_stats)
    
    daily_stats = pd.DataFrame({**general_stats, **time_range_stats}, index=[0])
    return daily_stats.astype(np.float16)
    
def daily_stats_basal(data, times, label_data='basal'):
    daily_data = pd.DataFrame({'datetime': times, 'value': data})
    daily_data.set_index('datetime', inplace=True)
    
    general_stats = calculate_stats(daily_data.values[:,0])
    general_stats = extract_stats(general_stats, label_data)
    mean_crossings = calculate_crossings(daily_data.values[:,0])
    general_stats[f'{label_data}_mean_crossings'] = mean_crossings
    masks = {
        'night': (daily_data.index.hour >= 0) & (daily_data.index.hour < 6),
        'morning': (daily_data.index.hour >= 6) & (daily_data.index.hour < 12),
        'afternoon': (daily_data.index.hour >= 12) & (daily_data.index.hour < 18),
        'evening': (daily_data.index.hour >= 18) & (daily_data.index.hour < 24),
        'late_evening': (daily_data.index.hour >= 22) & (daily_data.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(daily_data, mask)
        period_stats = extract_stats(period_stats, f'{label_data}_{period}')
        time_range_stats.update(period_stats)
    
    daily_stats = pd.DataFrame({**general_stats, **time_range_stats}, index=[0])
    return daily_stats.astype(np.float16)
    
def daily_stats_bolus(data, times, label_data='bolus'):
    daily_data = pd.DataFrame({'datetime': times, 'value': data})
    daily_data.set_index('datetime', inplace=True)
    
    general_stats = calculate_stats(daily_data.values[:,0])
    general_stats = extract_stats(general_stats, label_data)
    mean_crossings = calculate_crossings(daily_data.values[:,0])
    general_stats[f'{label_data}_mean_crossings'] = mean_crossings
    masks = {
        'night': (daily_data.index.hour >= 0) & (daily_data.index.hour < 6),
        'afternoon': (daily_data.index.hour >= 12) & (daily_data.index.hour < 18),
        'evening': (daily_data.index.hour >= 18) & (daily_data.index.hour < 24)
    }
    time_range_stats = {}
    for period, mask in masks.items():
        period_stats = calculate_stats(daily_data, mask)
        period_stats = extract_stats(period_stats, f'{label_data}_{period}')
        time_range_stats.update(period_stats)
    
    daily_stats = pd.DataFrame({**general_stats, **time_range_stats}, index=[0])
    return daily_stats.astype(np.float16)
    
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=stats.entropy(probabilities)
    return entropy.round(3)

def calculate_crossings(list_values):
    mean_crossing_indices = np.nonzero(np.diff(list_values > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return no_mean_crossings


def compute_cwt(data, scales, dt=1./12, normalize = True, waveletname = 'mexh', label_data = 'CGM ' ):
    times = data.index
    N = len(data)
    if waveletname in ['cmor', 'shan']:
        waveletname += '1.5-1'
        
    [coefficients_mex, frequencies_mex] = pywt.cwt(data.values.squeeze(), scales, waveletname, dt)
    [coefficients_morl, frequencies_morl] = pywt.cwt(data.values.squeeze(), 4*scales, 'morl', dt)
    # period_mex = 1. / frequencies_mex
    # period_morl = 1. / frequencies_morl
    # fft = np.fft.fft(data)
    # fftfreqs = np.fft.fftfreq(N, dt)
    scaleMatrix = np.ones([1, N]) * scales[:, None]

    power_mex =  (abs(coefficients_mex)) ** 2 / scaleMatrix
    power_morl =  (abs(coefficients_morl)) ** 2 / scaleMatrix

    # signed_log_power_mex = np.sign(coefficients_mex) * np.log2(power_mex)
    # signed_log_power_morl = np.sign(coefficients_morl) * np.log2(power_morl)
    # Replace -inf with the minimum finite value
    # signed_log_power_mex = np.where(np.isfinite(signed_log_power_mex), signed_log_power_mex, np.min(signed_log_power_mex[np.isfinite(signed_log_power_mex)])/10)
    # signed_log_power_morl = np.where(np.isfinite(signed_log_power_morl), signed_log_power_morl, np.min(signed_log_power_morl[np.isfinite(signed_log_power_morl)])/10)
    return (
        coefficients_mex,
        power_mex,
        # signed_log_power_mex.astype(np.float32),
        coefficients_morl,
        power_morl
        # signed_log_power_morl.astype(np.float32)
        # fft.astype(np.float32),
        # fftfreqs.astype(np.float32)
    )

# Apply median imputation to the features
def median_impute_features(train_features, val_features=None, test_features=None):
   """
   Impute missing values in features using median values from training data only.
   Imputation is performed feature-wise (column-wise).
   
   Args:
       train_features: Training features array (with potential NaNs)
       val_features: Validation features array (optional)
       test_features: Test features array (optional)
       
   Returns:
       Tuple of imputed feature arrays
   """
   # Make copies to avoid modifying originals
   imputed_train = np.copy(train_features)
   
   # Create imputed versions of validation and test if provided
   imputed_val = np.copy(val_features) if val_features is not None else None
   imputed_test = np.copy(test_features) if test_features is not None else None
   
   # Perform feature-wise (column-wise) imputation
   for col in range(train_features.shape[1]):
       # Calculate median for this specific feature from training data only
       median_value = np.nanmedian(train_features[:, col])
       
       # Replace NaNs with the training-derived median in training data
       mask_train = np.isnan(imputed_train[:, col])
       imputed_train[mask_train, col] = median_value
       
       # If validation data is provided, apply the same training-derived median
       if imputed_val is not None:
           mask_val = np.isnan(imputed_val[:, col])
           imputed_val[mask_val, col] = median_value
       
       # If test data is provided, apply the same training-derived median
       if imputed_test is not None:
           mask_test = np.isnan(imputed_test[:, col])
           imputed_test[mask_test, col] = median_value
           
   imputed_train = np.nan_to_num(imputed_train, nan=0.0)
   return imputed_train, imputed_val, imputed_test
    
def prepare_datasets(file_path):
    """
    Loads and processes dataset from the given file path.
    Filters data, extracts relevant features and labels, applies normalization, 
    and prepares input matrices for analysis.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        tuple: matrices, daily_features, individual_features, labels, participant_ids
    """
    
    # Load data
    with open(file_path, "rb") as f:
        data_dict = dill.load(f)
    
    # Filter data
    if len(data_dict)>350:
        filtered_data_dict = {}
        for participant_id, days_data in data_dict.items():
            filtered_days_data = [day_data for current_date, day_data in days_data.items()
                                  if (day_data['len_original_cgm'] > 252 and 
                                      np.sum(day_data['CGM_stats_daily_original'].isna()).sum() < 1)]
            if filtered_days_data:
                filtered_data_dict[participant_id] = filtered_days_data
    else:
        filtered_data_dict = data_dict
    
    del data_dict
    # Initialize data structures
    participant_ids = []
    X_features_individual, X_features_daily = [], []
    X_matrices_power = []
    labels = { 
        "hypo_early_night": [], "hypo_night": [], "hypo_long_night": [],
        "hypo_night_morning": [], "hyper_day": [], "hypo_late_night": [],
        "hyper_night": [], "hyper_early_night": [], "hypo_morning": [],
        "hypo_day": []
    }
    
    # Process data
    for participant_id, days_data in filtered_data_dict.items():
        participant_ids.extend([participant_id] * len(days_data))
        for day_data in days_data:
            X_matrices_power.append(day_data['full_matrix_cgm_power'])
            X_features_daily.append(np.concatenate([
                day_data['CGM_stats_daily_original'],
                day_data['Basal_stats_daily'],
                day_data['Carbs_stats_daily'],
                day_data['HR_stats_daily']], axis=1))
            X_features_individual.append(np.concatenate([
                [[day_data['A1c']]],
                [[day_data['Weight']]],
                [[day_data['Height']]],
                day_data['CGM_stats_participant'],
                day_data['Basal_stats_participant'],
                day_data['HR_stats_participant']], axis=1))
            
            for label in labels.keys():
                labels[label].append(day_data['next_day_cgm_labels_original'][label])
    
    # Convert lists to numpy arrays
    participant_ids = np.array(participant_ids)
    X_matrices_power = np.array(X_matrices_power)
    X_features_daily = np.array(X_features_daily)
    X_features_individual = np.array(X_features_individual)
    for key in labels:
        labels[key] = np.array(labels[key], dtype=np.float32)
    
    # Standardization function
    def standard_scale(data):
        scaler = StandardScaler()
        data[np.isinf(data)] = np.nan  # Replace infinities with NaN
        return scaler.fit_transform(data.reshape(-1, data.shape[-1]))
    
    # Apply standardization
    individual_features = standard_scale(X_features_individual)
    daily_features = standard_scale(X_features_daily)
    
    del filtered_data_dict, X_features_daily, X_features_individual    
    # Manual standard scaling for matrices
    def manual_standard_scaling(matrix):
        return matrix / np.nanstd(matrix)
    
    matrices = np.zeros_like(X_matrices_power)
    for i in range(X_matrices_power.shape[0]):
        for channel in range(3):  # Assuming 3 channels in last dimension
            matrices[i, :, :, channel] = manual_standard_scaling(X_matrices_power[i, :, :, channel])
    
    del X_matrices_power
    gc.collect()
    
    return matrices, daily_features, individual_features, labels, participant_ids


DEXI_folder = '/home/ma98/Datasets/Compressed_DEXI_Data/'

T1DEXI_CGM_Dataset = pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_Dataset.csv')) 
T1DEXI_CGM_Dataset['LBDTC'] = pd.to_datetime(T1DEXI_CGM_Dataset['LBDTC'])

T1DEXI_basal = pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_Insulin_Dataset.csv'))  
T1DEXI_basal['FADTC'] = pd.to_datetime(T1DEXI_basal['FADTC'], format='%Y-%m-%d %H:%M:%S')

T1DEXI_Reqcue_Carbs_Dataset = pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_Reqcue_Carbs_Dataset.csv'))  
T1DEXI_Reqcue_Carbs_Dataset['MLDTC'] = pd.to_datetime(T1DEXI_Reqcue_Carbs_Dataset['MLDTC'])

FAMLPM_Dataset = pd.read_parquet(os.path.join(DEXI_folder, 'FAMLPM_Dataset.csv')) 
FAMLPM_Dataset['FADTC'] = pd.to_datetime(FAMLPM_Dataset['FADTC'], format='%Y-%m-%d %H:%M:%S')

T1DEXI_sleep_Dataset = pd.read_parquet(os.path.join(DEXI_folder, 'T1DEXI_sleep_Dataset.csv')) 

T1DEXI_VS_Dataset = pd.read_csv(os.path.join(DEXI_folder, 'T1DEXI_VS_Dataset.csv.gz'))  
T1DEXI_VS_Dataset['VSDTC'] = pd.to_datetime(T1DEXI_VS_Dataset['VSDTC'], format='%Y-%m-%d %H:%M:%S')

shared_participants = sorted(set(T1DEXI_CGM_Dataset['USUBJID']) & set(T1DEXI_basal['USUBJID'])&
                             set(FAMLPM_Dataset['USUBJID']) & set(T1DEXI_VS_Dataset['USUBJID'])) 

print(len(shared_participants))

filtered_data_dict = {}
for participant_id, days_data in data_dict.items():
    filtered_days_data = []
    for current_date, day_data in days_data.items():

        if (day_data['len_original_cgm'] > 252 and np.sum(day_data['CGM_stats_daily_original'].isna()).sum()<1):
            filtered_days_data.append(day_data)
    
    if filtered_days_data:
        filtered_data_dict[participant_id] = filtered_days_data

with open('/home/ma98/data_generated/Foundation/Adult_DEXI.dill', 'wb') as f:
    dill.dump(filtered_data_dict, f)
    
print(len(data_dict),len(filtered_data_dict))

# Prepare the groups based on participant IDs
participant_ids = []
X_features_individual_cgm, X_features_daily_cgm = [], []
X_features_individual, X_features_daily = [], []
X_matrices_power = []
y_labels = { 
    "hypo_early_night": [], "hypo_night": [], "hypo_long_night": [],
    "hypo_night_morning": [], "hyper_day": [], "hypo_late_night": [],
    "hyper_night": [], "hyper_early_night": [], "hypo_morning": [],
    "hypo_day": []
}

for participant_id, days_data in filtered_data_dict.items():
    participant_ids.extend([participant_id] * len(days_data))
    for day_data in days_data:
        X_matrices_power.append(day_data['full_matrix_cgm_power'])
        X_features_daily_cgm.append(day_data['CGM_stats_daily_original'])
        X_features_individual_cgm.append(day_data['CGM_stats_participant'])
        X_features_daily.append(np.concatenate([
            day_data['CGM_stats_daily_original'],
            day_data['Basal_stats_daily'],
            day_data['Carbs_stats_daily'],
            day_data['HR_stats_daily']], axis=1))
        X_features_individual.append(np.concatenate([
            [[day_data['A1c']]],
            [[day_data['Weight']]],
            [[day_data['Height']]],
            day_data['CGM_stats_participant'],
            day_data['Basal_stats_participant'],
            day_data['HR_stats_participant']], axis=1))
    
        # Store labels in the dictionary
        for label in y_labels.keys():
            y_labels[label].append(day_data['next_day_cgm_labels_original'][label])
    
participant_ids = np.array(participant_ids)
X_matrices_power = np.array(X_matrices_power)
X_features_daily_cgm = np.array(X_features_daily_cgm)
X_features_individual_cgm = np.array(X_features_individual_cgm)
X_features_daily = np.array(X_features_daily)
X_features_individual = np.array(X_features_individual)

# Convert labels to NumPy arrays
for key in y_labels:
    y_labels[key] = np.array(y_labels[key], dtype=np.float32)


# Apply vectorized scaling
X_features_individual_cgm_scaled = standard_scale(X_features_individual_cgm)
X_features_daily_cgm_scaled = standard_scale(X_features_daily_cgm)
X_features_individual_scaled = standard_scale(X_features_individual)
X_features_daily_scaled = standard_scale(X_features_daily)


print('NaNs in the Data ...')
check_nans(X_matrices_power, "X_matrices_power")
check_nans(X_features_daily_cgm_scaled, "X_features_daily_cgm")
check_nans(X_features_individual_cgm_scaled, "X_features_individual_cgm")

