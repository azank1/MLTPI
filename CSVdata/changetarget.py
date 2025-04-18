import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def load_data(csv_filename="target.csv"):
    filepath = os.path.join("CSVdata", csv_filename)
    df = pd.read_csv(filepath)
    df['DateTime'] = pd.to_datetime(df['time'], unit="s")
    df.sort_values('DateTime', inplace=True)
    df.set_index('DateTime', inplace=True)
    df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
    return df

def plot_isp(df, title="Current ISP Signal"):
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['manual_signal'], label="ISP Signal", marker="o", linestyle="-")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Manual Signal")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def update_signal(df, start_date, end_date, intended_signal):
    """
    Updates the manual_signal column between start_date and end_date to the intended_signal.
    Then, forward-fills the updated column.
    """
    # Convert strings to datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Update the manual_signal column for the specified date range
    df.loc[start:end, 'manual_signal'] = intended_signal
    
    # Forward-fill the column to ensure continuity
    df['manual_signal'] = df['manual_signal'].ffill().fillna(0)
    return df

def main():
    # Load original data
    df = load_data("target.csv")
    
    # # Plot the current ISP
    # plot_isp(df, title="Current ISP Signal")
    
    # Get user inputs
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()
    intended_signal = int(input("Enter intended signal (0 or 1): ").strip())
    
    # Update the ISP signal in the specified date range
    df_updated = update_signal(df.copy(), start_date, end_date, intended_signal)
    
    # Plot updated ISP for review
    # plot_isp(df_updated, title="Updated ISP Signal")
    
    # Save the updated DataFrame as a new CSV
    output_filepath = os.path.join("CSVdata", "target.csv")
    df_updated.to_csv(output_filepath)
    print(f"Updated file saved as: {output_filepath}")

if __name__ == "__main__":
    main()
