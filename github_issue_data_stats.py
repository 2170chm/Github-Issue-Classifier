import pandas as pd
import matplotlib.pyplot as plt

data_path = 'processed_data.csv'

def load_data(file_path):
    return pd.read_csv(file_path)

def print_dataset_metrics(df):
    print("Basic stats of the dataset:")
    print("Number of entries:", df.shape[0])
    print("Number of features:", df.shape[1])
    print("\nDescriptive statistics:")
    print(df.describe())

def get_class_counts(df):
    return df['label'].value_counts()

def plot_class_distribution(class_counts):
    plt.figure(figsize=(10, 5))
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()
     
if __name__ == "__main__":
    df = load_data(data_path)
    
    encode_dict = {0: "bug", 1: "feature", 2: "question"}
    df['label'] = df['label'].replace(encode_dict)
    
    print_dataset_metrics(df)
    
    class_counts = get_class_counts(df)
    print("Class counts:\n", class_counts)

    plot_class_distribution(class_counts)
