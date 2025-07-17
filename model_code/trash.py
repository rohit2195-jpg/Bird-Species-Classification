import openml

# download dataset with DATASET_ID. DATASET_ID is OpenML ID
dataset = openml.datasets.get_dataset(DATASET_ID)

# display dataset info
print(dataset.name)