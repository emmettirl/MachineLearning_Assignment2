from copyreg import constructor

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import multiprocessing

from multiprocessing import Pool


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

num_samples=1000

########################################################################################################################
# Task 1:  pre-processing and visualisation
########################################################################################################################

def task1(df):

    target_df = df[df.iloc[:, 0].isin(TARGET_LABELS.keys())]

    labels, features = separate_labels_and_features(target_df)

    unique_labels = labels.unique()

    for label in unique_labels:
        first_instance = features[labels == label].iloc[0]
        display_image(label, first_instance)

    print('Unique labels:', unique_labels)
    print('Number of instances:', len(features))
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
# Task 2:  Evaluation Procedure
########################################################################################################################

def task2(labels, features, classifier, num_samples=None):
    num_processes = multiprocessing.cpu_count() - 2 # Use all but two cores to avoid overloading the system

    if num_samples:
        features = features.sample(n=num_samples, random_state=42)
        labels = labels.loc[features.index]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    y_tests = []
    y_preds = []

    pool = multiprocessing.Pool(processes=num_processes)
    args = [(train_index, test_index, features, labels, classifier) for train_index, test_index in kf.split(features)]
    fold_results = pool.map(train_and_evaluate_fold, args)
    pool.close()
    pool.join()

    start_time = time.time()
    total_folds = len(fold_results)

    for i, (accuracy, training_time, prediction_time, y_test, y_pred) in enumerate(fold_results):
        results.append((accuracy, training_time, prediction_time))
        y_tests.append(y_test)
        y_preds.append(y_pred)
        print(f'Fold {i + 1}: Accuracy: {accuracy:.2f}, Training Time: {training_time:.2f}s, Prediction Time: {prediction_time:.2f}s')


    # print confusion matrix
    OutputFormat.print_divider('h2')
    print('Confusion Matrix\n')
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))

    # print average results
    OutputFormat.print_divider('h2')

    results_df = pd.DataFrame(results, columns=['Accuracy', 'Training Time', 'Prediction Time'])

    summary_df = pd.DataFrame({
        'Min': results_df.min(),
        'Max': results_df.max(),
        'Average': results_df.mean()
    })

    print('Summary Results\n')
    print(summary_df)

    return results_df

def train_and_evaluate_fold(args):
    train_index, test_index, features, labels, classifier = args
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

    start_time = time.time()
    classifier.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = classifier.predict(X_test)
    prediction_time = time.time() - start_time - training_time

    accuracy = (y_pred == y_test).mean()
    return accuracy, training_time, prediction_time, y_test, y_pred

########################################################################################################################
# Task 3:  Perceptron Classifier
########################################################################################################################

def task3(labels, features, num_samples=None):
    classifier = Perceptron()
    results_df = task2(labels, features, classifier, num_samples)

########################################################################################################################
# Task 4:  Decision Tree Classifier
########################################################################################################################

def task4(labels, features, num_samples=None):
    classifier = DecisionTreeClassifier()
    results_df = task2(labels, features, classifier, num_samples)

########################################################################################################################
# Task 5:  K-Nearest Neighbours Classifier
########################################################################################################################

def task5(labels, features, num_samples=None, n_neighbors = 5):
    classifier = KNeighborsClassifier(n_neighbors)
    results_df = task2(labels, features, classifier, num_samples)


########################################################################################################################
# Task 6:  Support Vector Machine Classifier
########################################################################################################################

def task6(labels, features, num_samples=None):
    classifier = SVC()
    results_df = task2(labels, features, classifier, num_samples)

########################################################################################################################
# Task 7:  Classifier Comparison
########################################################################################################################

def task7():
    pass

########################################################################################################################
# Output Formatting
########################################################################################################################
class OutputFormat:
    SECTION_DIVIDER = {
        'title': '▀',
        'h1': '#',
        'h2': '*',
        'h3': '-'
    }

    def __init__(self):
        pass

    @staticmethod
    def print_header(section, text):
        print('\n' + OutputFormat.SECTION_DIVIDER[section] * 80)
        print(text.center(80, ' '))
        print(str(OutputFormat.SECTION_DIVIDER[section] * 80) + '\n')

    @staticmethod
    def print_divider(section):
        print("\n" + OutputFormat.SECTION_DIVIDER[section] * 80 + "\n")

    @staticmethod
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

    OutputFormat.print_header('title', 'Machine Learning Assignment 2')

    start_time = time.time()  # start measuring time

    # Create a cache folder if it does not exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    original_data = read_data()
    df = original_data.copy()

    OutputFormat.print_header('h1', 'Task 1: Pre-processing and Visualisation')
    labels, features = task1(df)


    OutputFormat.print_header('h1', 'Task 2: Evaluation Procedure')
    test_classifier = LogisticRegression(max_iter=1000)
    task2(labels, features, test_classifier, num_samples)

    OutputFormat.print_header('h1', 'Task 3: Perceptron Classifier')
    task3(labels, features, num_samples)

    OutputFormat.print_header('h1', 'Task 4: Decision Tree Classifier')
    task4(labels, features, num_samples)

    OutputFormat.print_header('h1', 'Task 5: K-Nearest Neighbours Classifier')
    task5(labels, features, num_samples, 5)

    OutputFormat.print_header('h1', 'Task 6: Support Vector Machine Classifier')
    task6(labels, features, num_samples)

    OutputFormat.print_header('h1', 'Task 7: Classifier Comparison')
    task7()

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