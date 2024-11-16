import threading

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import os
import multiprocessing
import pickle
import time
import pandas as pd
import matplotlib.pyplot as plt

source_filename = 'fashion-mnist_train'
file_extension = '.csv'
main_cache_filename = source_filename + ".pkl"

cache_folder = 'cache'

TARGET_LABELS = {
    5: 'Sandal',
    7: 'Sneaker',
    9: 'Ankle_boot'
}

num_samples=1800

########################################################################################################################
# Task 1:  pre-processing and visualisation
########################################################################################################################

def task1(df):
    OutputFormat.print_header('h1', 'Task 1: Pre-processing and Visualisation')

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

def task2(labels, features, num_samples=None):
    OutputFormat.print_header('h1', 'Task 2: Evaluation Procedure')

    test_classifier = LogisticRegression(max_iter=1000)
    results_df = run_classifier(labels, features, test_classifier, num_samples)

    # print average results
    OutputFormat.print_divider('h2')

    print('Summary Results')
    print(summary_results(results_df))


def run_classifier(labels, features, classifier, num_samples=None):

    if num_samples:
        features = features.sample(n=num_samples, random_state=42)
        labels = labels.loc[features.index]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    y_tests = []
    y_preds = []

    num_processes = multiprocessing.cpu_count() - 2 # Use all but two cores to avoid overloading the system
    pool = multiprocessing.Pool(processes=num_processes)
    args = [(train_index, test_index, features, labels, classifier) for train_index, test_index in kf.split(features)]
    fold_results = pool.map(train_and_evaluate_fold, args)
    pool.close()
    pool.join()

    for i, (accuracy, training_time, prediction_time, y_test, y_pred) in enumerate(fold_results):
        results.append((accuracy, training_time, prediction_time))
        y_tests.append(y_test)
        y_preds.append(y_pred)

        OutputFormat.print_divider('h2')
        print(f'Fold {i + 1}: Accuracy: {accuracy:.2f}, Training Time: {training_time:.2f}s, Prediction Time: {prediction_time:.2f}s')
        OutputFormat.print_divider('h3')

        confusion_matrix(y_test, y_pred)


    results_df = pd.DataFrame(results, columns=['Accuracy', 'Training Time', 'Prediction Time'])
    return results_df


def train_and_evaluate_fold(args):
    train_index, test_index, features, labels, classifier = args
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    start_time = time.time()
    classifier.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    y_pred = classifier.predict(X_test_scaled)
    prediction_time = time.time() - start_time - training_time

    accuracy = (y_pred == y_test).mean()
    return accuracy, training_time, prediction_time, y_test, y_pred


def confusion_matrix(y_test, y_pred):
    print('Confusion Matrix\n')
    print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']), '\n')


def summary_results(results):
    summary_df = pd.DataFrame({
        'Min': results.min(),
        'Max': results.max(),
        'Average': results.mean()
    })
    return summary_df

########################################################################################################################
# Task 3:  Perceptron Classifier
########################################################################################################################

def task3(labels, features, num_samples=None):
    OutputFormat.print_header('h1', 'Task 3: Perceptron Classifier')
    classifier_name = 'Perceptron'

    classifier = Perceptron()
    results_df = run_classifier(labels, features, classifier, num_samples)

    summary_df = summary_results(results_df)

    # print the average accuracy of the classifier
    print(f'Average Prediction Accuracy for {classifier_name}: {summary_df["Average"]["Accuracy"]:.2f}')

    plot_sample_size_vs_runtime(labels, features, classifier, classifier_name)


# run the classifier for different sample sizes to find the relationship between sample size and run time
def plot_sample_size_vs_runtime(labels, features, classifier, classifier_name):
    SampleSizeList = [2500, 5000, 7500, 10000, 12500, 15000, 17500]

    runtimes = []
    start_time = time.time()

    # run the classifier for different sample sizes to find the relationship between sample size and run time
    for new_num_samples in SampleSizeList:
        iteration_start_time = time.time()
        run_classifier(labels, features, classifier, new_num_samples)
        runtimes.append(time.time() - iteration_start_time)

        OutputFormat.progressbar(SampleSizeList.index(new_num_samples), len(SampleSizeList), start_time)

    # plot the relationship between sample size and run time
    plt.plot(SampleSizeList, runtimes)
    plt.xlabel('Sample Size')
    plt.ylabel('Run Time (s)')
    plt.title(f'Sample Size vs Run Time for {classifier_name} Classifier')
    plt.show()


########################################################################################################################
# Task 4:  Decision Tree Classifier
########################################################################################################################

def task4(labels, features, num_samples=None):
    OutputFormat.print_header('h1', 'Task 4: Decision Tree Classifier')
    classifier_name = 'Decision Tree'

    classifier = DecisionTreeClassifier()
    results_df = run_classifier(labels, features, classifier, num_samples)

    summary_df = summary_results(results_df)

    # print the average accuracy of the classifier
    print(f'Average Prediction Accuracy for {classifier_name}: {summary_df["Average"]["Accuracy"]:.2f}')

    plot_sample_size_vs_runtime(labels, features, classifier, classifier_name)

########################################################################################################################
# Task 5:  K-Nearest Neighbours Classifier
########################################################################################################################

def task5(labels, features, num_samples=None):
    OutputFormat.print_header('h1', 'Task 5: K-Nearest Neighbours Classifier')
    classifier_name = 'K-Nearest Neighbours'

    k = determine_best_k(labels, features, num_samples)
    classifier = KNeighborsClassifier(k)

    plot_sample_size_vs_runtime(labels, features, classifier, classifier_name)


def determine_best_k(labels, features, num_samples=None):
    min_k = 1
    max_k = 15
    results = []

    k = min_k
    while k <= max_k:
        classifier = KNeighborsClassifier(k)
        results_df = run_classifier(labels, features, classifier, num_samples)
        summary_df = summary_results(results_df)
        results.append(summary_df['Average']['Accuracy'])
        k += 1

    plt.plot(range(min_k, max_k + 1), results)
    plt.xlabel('K')
    plt.ylabel('Average Accuracy')
    plt.title('K vs Average Accuracy for K-Nearest Neighbours Classifier')
    plt.show()

    best_k = results.index(max(results)) + 1
    print(f'Best K: {best_k} at {max(results):.2f} accuracy')

    return best_k

########################################################################################################################
# Task 6:  Support Vector Machine Classifier
########################################################################################################################

def task6(labels, features, num_samples=None):
    OutputFormat.print_header('h1', 'Task 6: Support Vector Machine Classifier')

    best_params = determine_best_y_value_for_rbf(labels, features, num_samples)

    classifier = SVC(kernel='rbf', gamma=best_params['gamma'], C=best_params['C'])
    plot_sample_size_vs_runtime(labels, features, classifier, 'Support Vector Machine')

# https://scikit-learn.org/stable/modules/svm.html#parameters-of-the-rbf-kernel
def determine_best_y_value_for_rbf(labels, features, num_samples=None):
    if num_samples:
        features = features.sample(n=num_samples, random_state=42)
        labels = labels.loc[features.index]

    # Feature scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Define parameter grid
    # param_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'C': [0.1, 1, 10, 100, 1000]}
    param_grid = {'gamma': [0.001, 0.0025, 0.005, 0.01, 0.05, 0.75, 0.1, 1], 'C': [0.1, 1, 10, 100, 1000]}
    svc = SVC(kernel='rbf')

    # Grid search with cross-validation
    grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs= multiprocessing.cpu_count() - 1)
    grid_search.fit(features_scaled, labels)
    best_params = grid_search.best_params_

    # Plot heatmap of results for visualisation purposes
    plot_heatmap(grid_search.cv_results_, param_grid)
    print(f'Best parameters: {best_params}, with mean test score of {grid_search.best_score_:.2f}')

    return best_params

def plot_heatmap(results, param_grid):
    # Extract results
    mean_test_scores = results['mean_test_score'].reshape(len(param_grid['C']), len(param_grid['gamma']))

    # Plot heatmap using matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(mean_test_scores, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.xlabel('Gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(param_grid['gamma'])), param_grid['gamma'])
    plt.yticks(np.arange(len(param_grid['C'])), param_grid['C'])
    plt.title('Grid Search Mean Test Scores')
    plt.show()

########################################################################################################################
# Task 7:  Classifier Comparison
########################################################################################################################

def task7():
    OutputFormat.print_header('h1', 'Task 7: Classifier Comparison')

    pass

########################################################################################################################
# Output Formatting
########################################################################################################################
class OutputFormat:
    SECTION_DIVIDER = {
        'title': '█',
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

    labels, features = task1(df)
    task2(labels, features, num_samples)
    task3(labels, features, num_samples)
    task4(labels, features, num_samples)
    task5(labels, features, num_samples)
    task6(labels, features, num_samples)
    # task7()

    print(f'\nTotal Runtime: {time.time() - start_time:.2f}s')

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