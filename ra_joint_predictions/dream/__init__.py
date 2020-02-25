import logging
import numpy as np
import pandas as pd

def execute_dream_predictions():
	training_data = pd.read_csv('/train/training.csv')

	mean_preds = np.mean(training_data.iloc[:, 1:], axis = 0)

	val_dict = {}
	for idx, col in enumerate(training_data.columns[1:]):
		val_dict[col] = mean_preds[idx]

	pred_dicts = []

	template = pd.read_csv('/test/template.csv')
	for idx, row in template.iterrows():
    		pred_dict = val_dict.copy()
    		pred_dict['Patient_ID'] = row['Patient_ID']

    		pred_dicts.append(pred_dict)

	predictions = pd.DataFrame(pred_dicts, index = np.arange(len(pred_dicts)), columns = template.columns)
	predictions.to_csv('/output/predictions.csv', index = False)