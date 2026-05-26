import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OrdinalEncoder
import os

INPUT_DATA = r'c:\Users\trifo\OneDrive\Υπολογιστής\v0.1 - Antigrivity\OUTPUT\jira_with_timesheet_hours.csv'
OUTPUT_DATA = r'c:\Users\trifo\OneDrive\Υπολογιστής\v0.1 - Antigrivity\OUTPUT\training_data_v2.csv'

def engineer_features():
    print("Loading data...")
    df = pd.read_csv(INPUT_DATA)
    
    # Drop rows without actual timesheet hours or without assignee
    df = df.dropna(subset=['TS_timesheet_hours', 'JR_assignee'])
    
    # 1. Developer Historical Experience Features
    print("Engineering Developer Experience features...")
    # Calculate historical average hours per developer (excluding the current task to prevent data leakage in a real scenario, 
    # but for simplicity we calculate the overall mean or cumulative mean. Let's use overall mean for this baseline)
    dev_stats = df.groupby('JR_assignee')['TS_timesheet_hours'].agg(['mean', 'count']).reset_index()
    dev_stats.rename(columns={'mean': 'dev_avg_hours_historic', 'count': 'dev_tasks_count'}, inplace=True)
    
    df = df.merge(dev_stats, on='JR_assignee', how='left')
    
    # Project familiarity
    proj_fam = df.groupby(['JR_assignee', 'JR_project_key']).size().reset_index(name='dev_project_familiarity')
    df = df.merge(proj_fam, on=['JR_assignee', 'JR_project_key'], how='left')

    # 2. Text Embeddings
    print("Generating dense text embeddings...")
    # Fill NA summaries with empty strings
    df['JR_summary'] = df['JR_summary'].fillna('')
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    # E5 models require a prompt prefix context. We'll prepend 'query: ' for downstream asymmetric tasks
    texts_with_prefix = [f"query: {text}" for text in df['JR_summary'].tolist()]
    embeddings = model.encode(texts_with_prefix, show_progress_bar=True)
    emb_cols = [f'emb_{i}' for i in range(embeddings.shape[1])]
    df_emb = pd.DataFrame(embeddings, columns=emb_cols, index=df.index)
    
    # Concatenate embeddings to dataframe
    df = pd.concat([df, df_emb], axis=1)

    # 3. Categorical Encoding
    print("Encoding categorical variables...")
    cats = ['JR_project_key', 'JR_issue_type', 'JR_status']
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    # Fill NAs in categoricals
    for c in cats:
        df[c] = df[c].fillna('Unknown')
        
    df[[f"{c}_encoded" for c in cats]] = encoder.fit_transform(df[cats])

    # 4. Outlier Capping (IQR Method)
    print("Applying IQR capping on target variable...")
    Q1 = df['TS_timesheet_hours'].quantile(0.25)
    Q3 = df['TS_timesheet_hours'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    
    df['target_hours_capped'] = np.clip(df['TS_timesheet_hours'], a_min=0, a_max=upper_bound)
    
    print(f"Capped target upper bound: {upper_bound:.2f}")

    # Save final engineered dataset
    print(f"Saving engineered data to {OUTPUT_DATA}...")
    df.to_csv(OUTPUT_DATA, index=False)
    print("Done!")

if __name__ == '__main__':
    engineer_features()
