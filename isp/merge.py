import pandas as pd

def merge_csv_data(target_csv_path, new_csv_path):
    # Load existing target CSV and new TradingView CSV
    df_target = pd.read_csv(target_csv_path, parse_dates=['time'])
    df_new = pd.read_csv(new_csv_path, parse_dates=['time'])
    
    # Ensure the target CSV is sorted by time
    df_target.sort_values('time', inplace=True)
    
    # Identify new rows not present in target by time (assuming time uniquely identifies a candle)
    new_rows = df_new[~df_new['time'].isin(df_target['time'])]
    
    if not new_rows.empty:
        # Set default manual_signal = 0 for these new rows
        new_rows['manual_signal'] = 0
        # Append the new rows to the target and sort again
        df_updated = pd.concat([df_target, new_rows], ignore_index=True)
        df_updated.sort_values('time', inplace=True)
    else:
        df_updated = df_target.copy()
    
    return df_updated

# Example usage:
target_csv = "./CSVdata/target.csv"
new_csv = "./CSVdata/forwardtest.csv"
df_all = merge_csv_data(target_csv, new_csv)
df_all.to_csv(target_csv, index=False)  # Optionally save the merged CSV
