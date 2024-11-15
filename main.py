from copyreg import constructor

source_filename = 'fashion-mnist_train'
file_extension = '.csv'
main_cache_filename = source_filename + ".pkl"

cache_folder = 'cache'

import os
import pickle
import time

import pandas as pd
import matplotlib.pyplot as plt

TARGET_LABELS = {
    5: 'Sandal',
    7: 'Sneaker',
    9: 'Ankle_boot'
}

########################################################################################################################
# Task 1:  pre-processing and visualisation
########################################################################################################################

def task1(df):
    print_header('*', 'Task 1: Pre-processing and Visualisation')

    target_df = df[df.iloc[:, 0].isin(TARGET_LABELS.keys())]

    labels, features = separate_labels_and_features(target_df)

    unique_labels = labels.unique()

    for label in unique_labels:
        first_instance = features[labels == label].iloc[0]
        display_image(label, first_instance)
    return labels, features



def separate_labels_and_features(df):
    labels = df.iloc[:, 0]  # First column as labels
    features = df.iloc[:, 1:]  # All other columns as feature vectors
    return labels, features


def display_image(label, features):
    label = label
    pixels = features.values.reshape(28, 28)

    plt.figure()
    plt.title(f'Label: {TARGET_LABELS[label]}')
    plt.imshow(pixels, cmap='gray')
    plt.show()


########################################################################################################################
# Output Formatting
########################################################################################################################
def print_header(char, text):
    print('\n' + char * 80)
    print(text.center(80, ' '))
    print(str(char * 80) + '\n')


def print_divider(char):
    print("\n" + char * 80 + "\n")


# Function to display a progress bar so the user knows the program is still running and how far along it is
def progressbar(i, upper_range, start_time):
    # Calculate the percentage of completion
    percentage = (i / (upper_range - 1)) * 100
    # Calculate the number of '█' characters to display
    num_blocks = int(percentage/2)
    # Calculate elapsed time and estimated remaining time

    elapsed_time = time.time() - start_time
    if percentage > 0:
        estimated_total_time = elapsed_time / (percentage / 100)
        remaining_time = estimated_total_time - elapsed_time
    else:
        remaining_time = 0

    # Create the progress bar string
    progress_string = f'\r{("█" * num_blocks)}{("_" * (50 - num_blocks))} {percentage:.2f}% | Elapsed: {elapsed_time:.2f}s | Remaining: {remaining_time:.2f}s'
    if i == upper_range - 1:
        print(progress_string)
    else:
        print(progress_string, end='', flush=True)


########################################################################################################################
# Main
########################################################################################################################



def main():
    print_header('*', 'Machine Learning Project')

    start_time = time.time()  # start measuring time

    # Create a cache folder if it does not exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    original_data = read_data()
    df = original_data.copy()

    labels, features = task1(df)



def read_data():
    cache_path = os.path.join(cache_folder, main_cache_filename)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        print('Loaded data from cache\n')
    else:
        df = pd.read_csv(source_filename + file_extension, header=0)
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        print('Loaded data from csv file and cached it')
    return df

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()