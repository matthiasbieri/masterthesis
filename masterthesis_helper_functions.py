import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from datetime import timedelta
import random


def split_data_forecasting_window_with_validation(df, forecast_window, train_size=0.6, val_size=0.2, seed=None):
    """
    Splits the data into training, validation, and testing sets based on forecasting windows.

    Parameters:
    - df: pandas DataFrame. The DataFrame with a datetime index.
    - forecast_window: str. The forecasting window (e.g., '15min', '1h', '2h', '4h', '1d').
    - train_size: float. The proportion of the data to be used for training.
    - val_size: float. The proportion of the data to be used for validation.
    - seed: int, optional. Random seed for reproducibility.

    Returns:
    - train_df: pandas DataFrame. The training dataset.
    - val_df: pandas DataFrame. The validation dataset.
    - test_df: pandas DataFrame. The testing dataset.
    """

    # Ensure the index is of datetime type
    df.index = pd.to_datetime(df['datetime'])

    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Determine the number of observations per window
    num_obs_per_window = int(pd.Timedelta(forecast_window) / (df.index[1] - df.index[0]))

    # Group data into windows
    grouped = df.groupby(np.arange(len(df)) // num_obs_per_window)

    # Randomly select windows for training, validation, and testing
    all_windows = list(grouped.groups.keys())
    np.random.shuffle(all_windows)
    train_end = int(len(all_windows) * train_size)
    val_end = train_end + int(len(all_windows) * val_size)

    train_windows = all_windows[:train_end]
    val_windows = all_windows[train_end:val_end]
    test_windows = all_windows[val_end:]

    print("All Windows:", all_windows)
    print("Train Windows:", train_windows)
    print("Validation Windows:", val_windows)
    print("Test Windows:", test_windows)

    # Split the data
    train_df = pd.concat([grouped.get_group(w) for w in train_windows])
    val_df = pd.concat([grouped.get_group(w) for w in val_windows])
    test_df = pd.concat([grouped.get_group(w) for w in test_windows])

    return train_df, val_df, test_df


def plot_date_range(start_date, end_date, train_df, val_df, test_df):
    # Ensure the DataFrames are sorted by date
    train_df = train_df.sort_index()
    val_df = val_df.sort_index()
    test_df = test_df.sort_index()

    # Define a function to safely select data within a date range
    def safe_slice(df, start, end):
        mask = (df.index >= start) & (df.index <= end)
        return df.loc[mask]

    # Initialize the plot
    plt.figure(figsize=(18, 8))

    # Plotting for each DataFrame
    for df, label, color in zip([train_df, val_df, test_df], 
                                ['Training Set', 'Validation Set', 'Test Set'], 
                                ['blue', 'orange', 'green']):
        # Safely slice the DataFrame
        sliced_df = safe_slice(df, start_date, end_date)
        # Scatter plot
        plt.scatter(sliced_df.index, sliced_df['trafo_p_lv_mw'], label=label, color=color, alpha=0.5, s=10)

    # Add title and labels
    plt.title('Data Split for Forecasting ({0} - {1})'.format(start_date, end_date), fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Trafo P LV MW', fontsize=14)

    # Add gridlines and legend
    plt.grid(True)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=12, ncol=3)

    # Rotate date labels and adjust layout
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.show()


def evaluate_regression(y_true, y_pred, X):
    """
    Evaluates the performance of a regression model using various metrics.
    
    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values from the model
    - X: array-like, feature dataset used for prediction
    
    Returns:
    A dictionary containing the computed metrics.
    """
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    n = X.shape[0]  # Number of samples
    p = X.shape[1]  # Number of predictors
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mape = np.nanmean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100
    
    
    # Return results as a dictionary
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'MEDAE': medae,
        'RMSE': rmse,
        'R^2': r2,
        'Adjusted R^2': adjusted_r2,
        'MAPE': mape
    }
    
    return metrics


def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs. Predicted', xlabel='Actual', ylabel='Predicted'):
    """
    Plots the actual vs. predicted values of a regression model.
    
    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values from the model
    - title: str, optional. The title of the plot.
    - xlabel: str, optional. The x-axis label.
    - ylabel: str, optional. The y-axis label.
    """
    
    # Initialize the plot
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Line plot of best fit
    m, b = np.polyfit(y_true, y_pred, 1)
    plt.plot(y_true, m*y_true + b, color='red', label=f'Best Fit (R^2 = {r2_score(y_true, y_pred):.2f})')
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Add gridlines and legend
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Show the plot
    plt.show()


def plot_actual_vs_predicted_time_series(start_date, end_date, y_true, y_pred, title='Actual vs. Predicted', xlabel='Date', ylabel='Trafo P LV MW'):
    """
    Plots the actual vs. predicted values of a regression model for a given date range.
    
    Note: This function now assumes y_true is a pandas Series with a datetime index and y_pred is an array.
    """
    
    # Ensure y_true is sorted by date
    y_true_sorted = y_true.sort_index()
    
    # Slice y_true to the desired date range
    mask = (y_true_sorted.index >= start_date) & (y_true_sorted.index <= end_date)
    sliced_true = y_true_sorted.loc[mask]
    
    # Slice y_pred to match the length of sliced_true
    sliced_pred = y_pred[:len(sliced_true)]
    
    # Ensure the plotting uses the index from sliced_true for both actual and predicted values
    plt.figure(figsize=(18, 8))
    plt.plot(sliced_true.index, sliced_true, label='Actual', color='blue')
    plt.plot(sliced_true.index, sliced_pred, label='Predicted', color='red', linestyle='--')
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Plotting function for the residuals of a model (actual - predicted) vs. predicted values (y_pred)    
def plot_residuals_vs_predicted(y_true, y_pred, title='Residuals vs. Predicted', xlabel='Predicted', ylabel='Residuals'):
    """
    Plots the residuals vs. predicted values of a regression model.
    
    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values from the model
    - title: str, optional. The title of the plot.
    - xlabel: str, optional. The x-axis label.
    - ylabel: str, optional. The y-axis label.
    """
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Initialize the plot
    plt.figure(figsize=(10, 10))
    
    # Scatter plot
    plt.scatter(y_pred, residuals, alpha=0.5)
    
    # Add a horizontal line at y=0
    plt.axhline(0, color='red', linestyle='--')
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # Add gridlines
    plt.grid(True)
    
    # Show the plot
    plt.show()


# Adjusted function to create random intervals
def split_data_random_intervals(dataframe, start, end, min_days, max_days, seed=None):
    """
    Splits the data into chunks with random interval lengths between min_days and max_days.
    Returns a list of dataframes.

    Parameters:
    - dataframe: pandas DataFrame, the original dataset
    - start: datetime, the start date for the first interval
    - end: datetime, the end date for the last interval
    - min_days: int, minimum number of days for each interval
    - max_days: int, maximum number of days for each interval
    - seed: int or None, optional seed for random number generator for reproducibility

    Returns:
    A list of dataframes, each representing a random time interval.
    """
    # Set the random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)

    chunks = []
    current_start = start
    while current_start < end:
        interval = random.randint(min_days, max_days)
        current_end = current_start + timedelta(days=interval)
        
        # Make sure we don't go past the end of the data
        if current_end > end:
            current_end = end
            
        chunk = dataframe.loc[current_start:current_end]
        chunks.append(chunk)
        current_start = current_end  # Move to the end of the current interval for the next start
        
    return chunks