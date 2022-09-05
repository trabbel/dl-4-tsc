import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from scipy.interpolate import interp1d


def readucr(filename):
    data = np.loadtxt(filename)
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    try:
        os.makedirs(directory_path)
    except:
        # in case another machine created the path meanwhile !:(
        return None
    return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = f'{root_dir}/results/{classifier_name}/{archive_name}/'
    if os.path.exists(output_directory):
        return None
    os.makedirs(output_directory)
    return output_directory


def read_dataset(root_dir, archive_name, dataset_name):
    cur_root_dir = root_dir.replace('-temp', '')

    if archive_name == 'WP':
        # For test and train data based on the given structure
        root_dir_dataset = f'{cur_root_dir}/archives/{archive_name}/{dataset_name}/'
        df_train = pd.read_csv(
            f'{root_dir_dataset}/{dataset_name}_TRAIN.tsv',
            sep='\t',
            header=None,
        )


        df_test = pd.read_csv(
            f'{root_dir_dataset}/{dataset_name}_TEST.tsv',
            sep='\t',
            header=None,
        )


        y_train = df_train.values[:, 0]
        y_test = df_test.values[:, 0]


        x_train = df_train.drop(columns=[0])
        x_test = df_test.drop(columns=[0])

        x_train.columns = range(x_train.shape[1])
        x_test.columns = range(x_test.shape[1])

        x_train = x_train.values
        x_test = x_test.values

        # znorm
        std_ = x_train.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_train = (x_train - x_train.mean(axis=1, keepdims=True)) / std_

        std_ = x_test.std(axis=1, keepdims=True)
        std_[std_ == 0] = 1.0
        x_test = (x_test - x_test.mean(axis=1, keepdims=True)) / std_

    elif archive_name == 'WP_later':
        # Optional for later use
        file_name = f'{cur_root_dir}/archives/{archive_name}/{dataset_name}/{dataset_name}'

                
                

    else:
        # General case for .txt files
        file_name = f'{cur_root_dir}/archives/{archive_name}/{dataset_name}/{dataset_name}'

        x_train, y_train = readucr(f'{file_name}_TRAIN.txt')
        x_test, y_test = readucr(f'{file_name}_TEST.txt')
    return {
        dataset_name: (
            x_train.copy(),
            y_train.copy(),
            x_test.copy(),
            y_test.copy(),
        )
    }


def get_func_length(x_train, x_test, func):
    func_length = np.inf if func == min else 0
    n = x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])

    n = x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length


def transform_to_same_length(x, n_var, max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx = np.array(range(curr_length))
        idx_new = np.linspace(0, idx.max(), max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            f = interp1d(idx, ts, kind='cubic')
            new_ts = f(idx_new)
            ucr_x[i, :, j] = new_ts

    return ucr_x



def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                       columns=['precision', 'accuracy', 'recall', 'duration'])
    res['precision'] = precision_score(y_true, y_pred, average='macro')
    res['accuracy'] = accuracy_score(y_true, y_pred)

    if y_true_val is not None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val, y_pred_val)

    res['recall'] = recall_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res


def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=np.float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)


def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history[f'val_{metric}'])
    plt.title(f'model {metric}')
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def save_logs_t_leNet(output_directory, hist, y_pred, y_true, duration):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(f'{output_directory}history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration)
    df_metrics.to_csv(f'{output_directory}df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(f'{output_directory}df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, f'{output_directory}epochs_loss.png')


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(f'{output_directory}history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(f'{output_directory}df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])
    #print(row_best_model.keys)
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(f'{output_directory}df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
    plot_epochs_metric(hist, f'{output_directory}epochs_loss.png')

    return df_metrics
