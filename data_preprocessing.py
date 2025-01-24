import os
import pandas as pd

directory = ['C:/Master Thesis/Anomaly detection in manufacturing/Data/1_problem_Jun_24_2021_11-52-34/exp','C:/Master Thesis/Anomaly detection in manufacturing/Data/1_problem_Jun_24_2021_16-15-14/exp','C:/Master Thesis/Anomaly detection in manufacturing/Data/1_problem_Jun_24_2021_17-23-06/exp']

all_data = []
count=0
for i, item in enumerate(directory):
    for file_name in os.listdir(item):
        if file_name.endswith('breakdowns.csv'):  
            file_path = os.path.join(item, file_name)

            df = pd.read_csv(file_path)
            selected_columns = df[['ID', 'Tardiness', 'Overall_processing_time']]
            last_column_as_breaks = df.iloc[:, -1].rename('BREAKS')

            extracted_data = pd.concat([selected_columns, last_column_as_breaks], axis=1)
            
            extracted_data['Source_File'] = file_name
            
            
            all_data.append(extracted_data)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_csv(f'90data_{i + 1}.csv', index=False)
    all_data = []

directory = 'C:/Master Thesis/Anomaly detection implementation'
for file_name in os.listdir(directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory, file_name)
        
        df = pd.read_csv(file_path)
        
        extracted_data = df[['ID', 'Tardiness', 'Overall_processing_time', 'BREAKS']]
      
        
        extracted_data['Source_File'] = df['Source_File']
        
        all_data.append(extracted_data)

combined_data = pd.concat(all_data, ignore_index=True)

combined_data.to_csv('90combined_data_breaks.csv', index=False)