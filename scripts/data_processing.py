import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
from scripts.matrix_utils import compute_q_pi, preprocess_passing_matrix

def process_team_games(folder, team_name, max_games=38):
    # Initialize empty DataFrames for q and pi
    q_df = pd.DataFrame()
    pi_df = pd.DataFrame()

    # Define regex pattern to extract dates in YYYY-MM-DD format
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}')

    # List all files in the passing_matrix folder for the given team
    passing_matrix_dir = f"data/passing_matrix/{folder}/"
    all_files = os.listdir(passing_matrix_dir)

    # Extract unique dates from passing_matrix files
    dates = set()
    for filename in all_files:
        if team_name in filename:
            match = date_pattern.search(filename)
            if match:
                dates.add(match.group())

    # Convert the set to a sorted list (optional: sort by date)
    sorted_dates = sorted(list(dates), key=lambda date: datetime.strptime(date, "%Y-%m-%d"))

    # Limit to max_games if necessary
    sorted_dates = sorted_dates[:max_games]

    for date in sorted_dates:
        # Construct file paths using the extracted date
        pm_filename = f"{team_name}_passing_matrix_{date}.csv"
        alpha_filename = f"{team_name}_initial_possession_{date}.csv"

        pm_path = os.path.join(passing_matrix_dir, pm_filename)
        alpha_path = os.path.join(passing_matrix_dir, alpha_filename)

        # If either file doesn't exist, skip to the next date
        if not os.path.isfile(pm_path) or not os.path.isfile(alpha_path):
            print(f"Skipping date {date}: Missing files.")
            continue

        # Read CSVs
        P_df = pd.read_csv(pm_path)
        alpha_df = pd.read_csv(alpha_path)

        # Convert to numpy arrays
        # Assuming the first column is an ID or index column
        P_raw = P_df.iloc[:, 1:].to_numpy()
        alpha = alpha_df.iloc[:-3, 1].to_numpy()  # Exclude last 3 rows

        # Preprocess and compute q and pi
        P = preprocess_passing_matrix(P_raw)
        q, pi = compute_q_pi(P, alpha)

        # Extract player names from the DataFrame columns (assuming they are in the header)
        player_names = P_df.columns[1:-3]  # Adjust slicing if necessary

        # Build DataFrames for the current game with the date as the column name
        q_g_df = pd.DataFrame({date: q}, index=player_names)
        pi_g_df = pd.DataFrame({date: pi}, index=player_names)

        # Outer-join to preserve all players across games
        q_df = q_df.join(q_g_df, how='outer')
        pi_df = pi_df.join(pi_g_df, how='outer')

    # Optionally, set the DataFrame index name to 'Player' for clarity
    q_df.index.name = 'Player'
    pi_df.index.name = 'Player'

    return q_df, pi_df



def rolling_average(pi, lag=6, fill_mean=True, remove_threshold=0.01):
    pi_arr = np.asarray(pi, dtype=float)

    # remove values less than remove_threshold, usually because the player got negligible playing time.  can set to 0 if desired.
    removed_indices = np.where(pi_arr < remove_threshold)[0].tolist()
    pi_filtered = pi_arr[pi_arr >= remove_threshold]
    
    pi_new = np.full(len(pi_arr), np.nan)
    
    if pi_filtered.size == 0:
        return pi_new
    
    grand_mean = pi_filtered.mean()
    n = len(pi_filtered)
    
    # Compute cumulative sum
    cumsum = np.cumsum(np.insert(pi_filtered, 0, 0))
    
    # Initialize rolling averages array
    pi_values = np.empty(n)
    
    if fill_mean:
        # Fill missing values with grand_mean for initial positions
        for i in range(n):
            if i < lag - 1:
                missing = lag - 1 - i
                window_sum = cumsum[i+1] - cumsum[0] + missing * grand_mean
                pi_values[i] = window_sum / lag
            else:
                window_sum = cumsum[i+1] - cumsum[i - lag + 1]
                pi_values[i] = window_sum / lag
    else:
        # Use partial windows for initial positions
        window_sums = cumsum[lag:] - cumsum[:-lag]
        pi_values[lag - 1:] = window_sums / lag
        # Handle initial positions separately
        for i in range(lag - 1):
            pi_values[i] = (cumsum[i+1] - cumsum[0]) / (i+1)
    
    # Assign rolling averages back to pi_new
    kept_indices = [i for i in range(len(pi_arr)) if i not in removed_indices]
    pi_new[kept_indices] = pi_values
    
    return pi_new

