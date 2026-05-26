import pandas as pd
import glob
import os

JIRA_CSV = r'c:\Users\trifo\OneDrive\Υπολογιστής\v0.1 - Antigrivity\DATA\raw\JIRA\Jira - Web Bunch.csv'
TIMESHEETS_DIR = r'c:\Users\trifo\OneDrive\Υπολογιστής\v0.1 - Antigrivity\DATA\raw\Timesheets'
OUTPUT_DIR = r'c:\Users\trifo\OneDrive\Υπολογιστής\v0.1 - Antigrivity\DATA\processed'

def load_and_merge_data():
    print("Loading Jira data...")
    df_jira = pd.read_csv(JIRA_CSV, low_memory=False)
    
    print("\nLoading Timesheets...")
    ts_files = glob.glob(os.path.join(TIMESHEETS_DIR, "*.xlsx"))
    ts_dfs = []
    for f in ts_files:
        print(f"  Reading {os.path.basename(f)}...")
        df_ts = pd.read_excel(f)
        ts_dfs.append(df_ts)
        
    df_ts_all = pd.concat(ts_dfs, ignore_index=True)
    
    print(f"\nJira records: {len(df_jira)}")
    print(f"Timesheet records: {len(df_ts_all)}")
    
    # Analyze columns to identify keys for joining.
    print("\nJira columns snapshot:")
    print(df_jira.columns.tolist()[:15])
    
    print("\nTimesheet columns snapshot:")
    print(df_ts_all.columns.tolist())

if __name__ == "__main__":
    load_and_merge_data()
