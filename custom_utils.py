from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import math
import csv
import os


_VALID_DATASET = ['swda', 'mrda', 'dyda', 'maptask', 'osda', 'mastodon']
_OUTPUT_DIM = {
    'swda': 42,
    'mrda': 5,
    'dyda': 4,
    'maptask': 12,
    'osda': 42,
    'mastodon': 15
}
_FIGURE_SIZE = {
    'swda': (12, 6),
    'mrda': (8, 5),
    'dyda': (6, 5),
    'maptask': (9, 5),
    'osda': (12, 6),
    'mastodon': (10, 5),
    'default': (8, 4)
}
_VALID_WORD2VEC = ['charngram.100d',
                   'fasttext.en.300d',
                   'fasttext.simple.300d',
                   'glove.42B.300d',
                   'glove.840B.300d',
                   'glove.twitter.27B.25d',
                   'glove.twitter.27B.50d',
                   'glove.twitter.27B.100d',
                   'glove.twitter.27B.200d',
                   'glove.6B.50d',
                   'glove.6B.100d',
                   'glove.6B.200d',
                   'glove.6B.300d']

_label_percentages = {}
_figure_width_rate = 0.25
_figure_height_rate = 0.15


# Get the calculated percentages of label.
def get_label_percentages():
    return _label_percentages


# Judge if the given dataset is valid.
def check_valid_dataset(dataset_name: str):
    lower_dataset_name = dataset_name.lower()
    if not lower_dataset_name in _VALID_DATASET:
        print("Expected name of dataset: ")
        for expect_dataset_name in _VALID_DATASET:
            print("\t{}".format(expect_dataset_name))
        raise AssertionError('Given name of dataset \'{}\' is invalid. You should implement it in _VALID_DATASET in custom_utils.py first!'.format(dataset_name))


# Judge the output-dimension of dataset.
# You can implement information of new dataset HERE
def judge_output_dim_dataset(dataset_name: str):
    lower_dataset_name = dataset_name.lower()
    if lower_dataset_name in _OUTPUT_DIM.keys():
        return _OUTPUT_DIM[lower_dataset_name]
    else:
        raise AssertionError('You haven\'t implemented the output dimension of dataset \'{}\' in _OUTPUT_DIM in custom_utils.py.'.format(dataset_name))


# Process sequence to fixed length.
def _process_sequence(words: list, max_seq_len: int):
    if max_seq_len == -1:
        return words
    origin_len = len(words)
    if origin_len > max_seq_len:
        words = words[:max_seq_len]
    else:
        words += ['<PAD>' for _ in range(max_seq_len - origin_len)]
    return words


# Do sequence length analysis.
def analysis_seq_len(dataset_folder: str):
    '''
    :param dataset_folder: The folder that contains dataset files.
        There should be [train.txt, dev.txt, test.txt] in this folder (.txt could be ignored).
    '''

    if not os.path.exists(dataset_folder):
        print('Designated dataset folder [{}] could not be found, the program has make one automatically'.format(
            dataset_folder)
        )
        os.mkdir(dataset_folder)

    dataset_types = ['train', 'dev', 'test']

    total_seq_len = [0, 0, 0]
    total_seq_count = [0, 0, 0]
    index = -1

    for dataset_type in dataset_types:

        index += 1
        dataset_file = dataset_folder + dataset_type

        if not os.path.exists(dataset_file):
            dataset_file = dataset_folder + dataset_type + '.txt'
            if not os.path.exists(dataset_file):
                print('\'{}\' or \'{}\' file could not be found in folder \'{}\''.format(
                    dataset_type, dataset_file, dataset_folder
                ))
                continue

        rf = open(dataset_file, 'r', encoding='utf-8')
        for line in rf:
            total_seq_count[index] += 1
            total_seq_len[index] += len(line.rstrip().split(' ')) - 1

        rf.close()

    print('\n******************** Analysis result ********************')
    print('Average seq len in train file: {:.2f}'.format(total_seq_len[0] / total_seq_count[0]))
    print('Average seq len in dev file: {:.2f}'.format(total_seq_len[1] / total_seq_count[1]))
    print('Average seq len in test file: {:.2f}'.format(total_seq_len[2] / total_seq_count[2]))
    print('Average seq len in total: {:.2f}'.format(sum(total_seq_len) / sum(total_seq_count)))
    print('*********************************************************')


# Do sequence token complexity analysis.
def analysis_token_complexity(dataset_folder: str):
    '''
        :param dataset_folder: The folder that contains dataset files.
            There should be [train.txt, dev.txt, test.txt] in this folder (.txt could be ignored).
    '''

    if not os.path.exists(dataset_folder):
        print('Designated dataset folder [{}] could not be found, the program has make one automatically'.format(
            dataset_folder)
        )
        os.mkdir(dataset_folder)

    dataset_types = ['train', 'dev', 'test']

    token_counts = [0, 0, 0]
    token_types = [{}, {}, {}]
    total_token_types = {}
    index = -1

    for dataset_type in dataset_types:

        index += 1
        dataset_file = dataset_folder + dataset_type

        if not os.path.exists(dataset_file):
            dataset_file = dataset_folder + dataset_type + '.txt'
            if not os.path.exists(dataset_file):
                print('\'{}\' or \'{}\' file could not be found in folder \'{}\''.format(
                    dataset_type, dataset_file, dataset_folder
                ))
                continue

        rf = open(dataset_file, 'r', encoding='utf-8')
        for line in rf:
            tokens = line.rstrip().split(' ')[1:]
            token_counts[index] += len(tokens)

            for token in tokens:
                if token not in token_types[index].keys():
                    token_types[index][token] = 1
                else:
                    token_types[index][token] += 1

                if token not in total_token_types.keys():
                    total_token_types[token] = 1
                else:
                    total_token_types[token] += 1

        rf.close()

    print('\n******************** Analysis result ********************')
    print('Token complexity in train file: {} different tokens in all {}, average {:.2f} types per 100000 tokens.'.format(
        len(token_types[0]), token_counts[0], len(token_types[0]) / token_counts[0] * 100000
    ))
    print('Token complexity in dev file: {} different tokens in all {}, average {:.2f} types per 100000 tokens.'.format(
        len(token_types[1]), token_counts[1], len(token_types[1]) / token_counts[1] * 100000
    ))
    print('Token complexity in test file: {} different tokens in all {}, average {:.2f} types per 100000 tokens.'.format(
        len(token_types[2]), token_counts[2], len(token_types[2]) / token_counts[2] * 100000
    ))
    print('Token complexity in total: {} different tokens in all {}, average {:.2f} types per 100000 tokens.'.format(
        len(total_token_types), sum(token_counts), len(total_token_types) / sum(token_counts) * 100000
    ))
    print('*********************************************************')


# Do label frequency analysis.
def analysis_label_freq(dataset_name: str, dataset_folder: str, figure_path: str):
    '''
        :param dataset_name: Name of dataset.
        :param dataset_folder: The folder that contains dataset files.
            There should be [train.txt, dev.txt, test.txt] in this folder (.txt could be ignored).
        :param figure_path: The path to write figure.
    '''

    if not os.path.exists(dataset_folder):
        print('Designated dataset folder [{}] could not be found, the program has make one automatically'.format(
            dataset_folder)
        )
        os.mkdir(dataset_folder)

    dataset_types = ['train', 'dev', 'test']

    all_labels = {}
    total_count = 0

    label_dim = 0
    index = -1

    for dataset_type in dataset_types:

        index += 1
        dataset_file = dataset_folder + dataset_type

        if not os.path.exists(dataset_file):
            dataset_file = dataset_folder + dataset_type + '.txt'
            if not os.path.exists(dataset_file):
                print('\'{}\' or \'{}\' file could not be found in folder \'{}\''.format(
                    dataset_type, dataset_file, dataset_folder
                ))
                continue

        rf = open(dataset_file, 'r', encoding='utf-8')
        for line in rf:
            label = int(line.rstrip().split(' ')[0])
            label_dim = label+1 if label+1 > label_dim else label_dim

            if not label in all_labels.keys():
                all_labels[label] = 1
            else:
                all_labels[label] += 1

            total_count += 1

        rf.close()

    x = [k for k in range(label_dim)]
    y = []

    print('\n******************** Analysis result ********************')
    for k in range(label_dim):
        if k in all_labels.keys():
            y.append(all_labels[k])
            print('Label {}\'s count is {}, {:.4f}% of total.'.format(
                k, all_labels[k], all_labels[k] / total_count * 100.0
            ))
        else:
            y.append(0)
            print('Label {} does not appear.'.format(k))
    print('*********************************************************')

    if dataset_name.lower() not in _FIGURE_SIZE.keys():
        dataset_name = 'default'
    plt.figure(figsize=_FIGURE_SIZE[dataset_name.lower()])

    plt.bar(x, y)
    plt.savefig(figure_path)


# Do label corresponding sequences' length analysis.
def analysis_label_seq_len(dataset_folder: str):
    '''
        :param dataset_folder: The folder that contains dataset files.
            There should be [train.txt, dev.txt, test.txt] in this folder (.txt could be ignored).
    '''

    if not os.path.exists(dataset_folder):
        print('Designated dataset folder [{}] could not be found, the program has make one automatically'.format(
            dataset_folder)
        )
        os.mkdir(dataset_folder)

    dataset_types = ['train', 'dev', 'test']
    label_lens = {}
    label_counts = {}
    label_dim = 0
    index = -1

    for dataset_type in dataset_types:

        index += 1
        dataset_file = dataset_folder + dataset_type

        if not os.path.exists(dataset_file):
            dataset_file = dataset_folder + dataset_type + '.txt'
            if not os.path.exists(dataset_file):
                print('\'{}\' or \'{}\' file could not be found in folder \'{}\''.format(
                    dataset_type, dataset_file, dataset_folder
                ))
                continue

        rf = open(dataset_file, 'r', encoding='utf-8')
        for line in rf:
            items = line.rstrip().split(' ')
            label = int(items[0])
            seq_len = len(items) - 1
            label_dim = label+1 if label+1 > label_dim else label_dim

            if not label in label_lens.keys():
                label_lens[label] = seq_len
                label_counts[label] = 1
            else:
                label_lens[label] += seq_len
                label_counts[label] += 1

        rf.close()

    counts = []
    avg_lens = []
    for k in range(label_dim):
        if k in label_counts.keys():
            counts.append(label_counts[k])
            avg_lens.append(label_lens[k] / label_counts[k])
        else:
            counts.append(0)
            avg_lens.append(0)


    print('\n******************** Analysis result ********************')
    for k in range(label_dim):
        print('Label {} has average sequence length of {:.2f} tokens.'.format(
            k, avg_lens[k]
        ))
    print('*********************************************************')


# Read experiment results from an .xls file.
def _read_xls_file(dataset_name, expected_dim, select_models,
                   xls_path='./experiment_result.xls', skip_len=3):

    from math import isnan

    xls = pd.ExcelFile(xls_path)
    dataset_names = xls.sheet_names

    assert dataset_name in dataset_names, \
        'Given dataset name {} could not be found in xls file. Expected: {}.'.format(
            dataset_name, dataset_names
        )

    sheet_index = dataset_names.index(dataset_name)
    sheet = xls.parse(sheet_index)

    results = {}
    valid_models = []
    model_names = sheet.columns.values

    for model_name in model_names:

        if model_name == 'Model name':
            continue

        if len(select_models) > 0 and model_name not in select_models:
            continue

        data_frame = sheet[model_name]
        results[model_name] = []

        for item in data_frame:
            if not isnan(item):
                results[model_name].append(item)

        if len(results[model_name]) >= skip_len + expected_dim:
            results[model_name] = results[model_name][skip_len: skip_len + expected_dim]
            valid_models.append(model_name)
        else:
            results[model_name] = []

    return results, valid_models


# Plot experimental results in a figure.
def plot_experimental_results(dataset_name: str, dataset_folder: str, figure_path: str, select_models: list,
                              xls_path='./experiment_result.xls', skip_len=3):

    if not os.path.exists(dataset_folder):
        print('Designated dataset folder [{}] could not be found, the program has make one automatically'.format(
            dataset_folder)
        )
        os.mkdir(dataset_folder)

    dataset_types = ['train', 'dev', 'test']

    all_labels = {}
    label_dim = 0
    total_count = 0

    for dataset_type in dataset_types:

        dataset_file = dataset_folder + dataset_type

        if not os.path.exists(dataset_file):
            dataset_file = dataset_folder + dataset_type + '.txt'
            if not os.path.exists(dataset_file):
                print('\'{}\' or \'{}\' file could not be found in folder \'{}\''.format(
                    dataset_type, dataset_file, dataset_folder
                ))
                continue

        rf = open(dataset_file, 'r', encoding='utf-8')
        for line in rf:
            label = int(line.rstrip().split(' ')[0])
            label_dim = label+1 if label+1 > label_dim else label_dim

            if not label in all_labels.keys():
                all_labels[label] = 1
            else:
                all_labels[label] += 1

            total_count += 1

        rf.close()

    results, valid_models = _read_xls_file(dataset_name, label_dim, select_models, xls_path, skip_len)

    x = [k for k in range(label_dim)]
    xnew = np.linspace(0, label_dim-1, label_dim * 8)
    y = []

    for k in range(label_dim):
        if k in all_labels.keys():
            y.append(all_labels[k] / total_count)
        else:
            y.append(0.00)

    if dataset_name.lower() not in _FIGURE_SIZE.keys():
        dataset_name = 'default'
    plt.figure(figsize=_FIGURE_SIZE[dataset_name.lower()])

    for valid_model in valid_models:
        model_result = results[valid_model]
        spl = make_interp_spline(x, model_result, k=1)
        model_smooth_result = spl(xnew)
        plt.axis([0, label_dim - 1, 0.0, 1.0])
        plt.plot(xnew, model_smooth_result)
    plt.legend(valid_models)

    plt.bar(x, y)
    plt.savefig(figure_path)


# Analysis label lists.
def _analysis_label(label_list: list, output_dim: int):

    label_num = {k: 0 for k in range(output_dim)}
    total_num = len(label_list)

    for label in label_list:
        label_ = int(label)
        label_num[label_] += 1

    label_num_sorted = {k: v for k, v in sorted(label_num.items())}
    for k in label_num_sorted.keys():
        print("Training amount of label[{}]:\t{} ({:.4f})%".format(
            k, label_num_sorted[k], label_num_sorted[k] / total_num
        ))
        _label_percentages[int(k)] = label_num_sorted[k] / total_num


# Convert label-text type dataset to .csv file.
def convert_dataset_to_csv(dataset_folder: str, output_dim: int, do_convert: bool, max_seq_len: int):
    '''
    :param dataset_folder: The folder that contains dataset files.
        There should be [train.txt, dev.txt, test.txt] in this folder (.txt could be ignored).
    :param output_dim: The num of label categories of your dataset.
    :param do_convert: Whether to convert original dataset files to .csv files. This is necessary if .csv datasets do not exist.
    :param max_seq_len: No used.
    :return:
    '''

    if not os.path.exists(dataset_folder):
        print('Designated dataset folder [{}] could not be found, the program has make one automatically'.format(
            dataset_folder)
        )
        os.mkdir(dataset_folder)

    dataset_types = ['train', 'dev', 'test']

    for dataset_type in dataset_types:

        auto_convert = False

        if os.path.exists(dataset_folder + dataset_type):
            y = []
            csv_rows = []
            rf = open(dataset_folder + dataset_type, 'r', encoding='utf-8')
            for line in rf:
                items = line.rstrip().split(' ')

                # Abandon invalid lines.
                if len(items) <= 1:
                    continue

                y.append(items[0])
                csv_rows.append([items[0], ' '.join(_process_sequence(items[1:], max_seq_len))])
            rf.close()

            # Specially analysis on train dataset.
            if dataset_type == 'train':
                _analysis_label(y, output_dim)

            if not os.path.exists(dataset_folder + dataset_type + '.csv'):
                print(
                    'File {}.csv could not be found in dataset folder [{}], program is trying to convert original data.'.format(
                        dataset_type, dataset_folder
                    ))
                auto_convert = True

            if do_convert or auto_convert:
                wf = open(dataset_folder + dataset_type + '.csv', 'w', encoding='utf-8', newline='')
                csv_wf = csv.writer(wf)
                csv_wf.writerow(['label', 'text'])
                csv_wf.writerows(csv_rows)
                print("Successfully convert {} to .csv file.".format(dataset_type))

        elif os.path.exists(dataset_folder + dataset_type + '.txt'):
            y = []
            csv_rows = []
            rf = open(dataset_folder + dataset_type + '.txt', 'r', encoding='utf-8')
            for line in rf:
                items = line.rstrip().split(' ')
                y.append(items[0])
                csv_rows.append([items[0], ' '.join(_process_sequence(items[1:], max_seq_len))])
            rf.close()

            # Specially analysis on train dataset.
            if dataset_type == 'train':
                _analysis_label(y)

            if not os.path.exists(dataset_folder + dataset_type + '.csv'):
                print(
                    'File {}.csv could not be found in dataset folder [{}], program is trying to convert original data.'.format(
                        dataset_type, dataset_folder
                    ))
                auto_convert = True

            if do_convert or auto_convert:
                wf = open(dataset_folder + dataset_type + '.csv', 'w', encoding='utf-8', newline='')
                csv_wf = csv.writer(wf)
                csv_wf.writerow(['label', 'text'])
                csv_wf.writerows(csv_rows)
                print("Successfully convert {} to .csv file.".format(dataset_type))

        else:
            raise OSError('Could not find file {} or {}.txt in given dataset folder [{}]'.format(
                dataset_type, dataset_type, dataset_folder
            ))


# Judge if the given pre-trained word2vec file is valid.
def check_valid_word2vec(word2vec_name: str):
    if not word2vec_name in _VALID_WORD2VEC:
        print("Expected pre-trained word2vec name: ")
        for expect_word2vec_name in _VALID_WORD2VEC:
            print("\t{}".format(expect_word2vec_name))
        raise AssertionError('Given pre-trained word2vec file [{}] is not valid.'.format(word2vec_name))


# Judge the dimension of specific pre-trained word2vec file.
def judge_dim_word2vec(word2vec_name: str):
    if '300' in word2vec_name:
        return 300
    elif '200' in word2vec_name:
        return 200
    elif '100' in word2vec_name:
        return 100
    elif '50' in word2vec_name:
        return 50
    elif '25' in word2vec_name:
        return 25
    else:
        raise AssertionError('Given pre-trained word2vec file [{}] is not valid.'.format(word2vec_name))


# Compute count, accuracy, precision, recall, F1-score of a single label.
def _compute_label_quotas(confusion_matrix: np.ndarray, index: int, num_total_samples: int):

    FP = np.sum(confusion_matrix[:, index]).item() - confusion_matrix[index][index]
    FN = np.sum(confusion_matrix[index]).item() - confusion_matrix[index][index]
    TP = confusion_matrix[index][index]
    TN = num_total_samples - FP - FN - TP

    # accuracy = (TP + TN) / num_total_samples
    accuracy = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0.0
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-4)
    cba = precision if precision < recall else recall

    # Imbalance accuracy metric (IAM).
    # The result of calculation is between -1 and 1.
    iam = (TP - FP) if FP > FN else (TP - FN)
    iam = (iam / (TP + FP)) if FP > FN else (iam / (TP + FN))
    if math.isnan(iam):
        iam = 0.0

    label_quotas = {'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN, 'count': TP + FN,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1_score,
                    'cba': cba,
                    'iam': iam,
                    'max_predicted': np.argmax(confusion_matrix[index]).item()}

    return label_quotas


# Get all quotas.
def get_matrix_quotas(confusion_matrix: np.ndarray):
    '''
    :param confusion_matrix: np.ndarray, must be of squared shape.
        Each element of confusion matrix should be the number of prediction.
        For example, x = confusion_matrix[i][j] represents there are x samples of label i but predicted to be label j.
    :return quotas: A dictionary that contains all necessary quotas.
    '''

    assert len(confusion_matrix.shape) == 2, \
        'Matrix to compute quotas on must be in 2 dimensions, got {}.'.format(len(confusion_matrix.shape))
    assert (confusion_matrix.shape[0] == confusion_matrix.shape[1]), \
        'Height and width of input matrix must be the same, got {} and {}.'.format(confusion_matrix.shape[0], confusion_matrix.shape[1])

    quotas = {
        'label_details': {},
        'total_accuracy': 0.0,
        'total_precision': 0.0,
        'total_recall': 0.0,
        'total_f1': 0.0,
        'total_cba': 0.0,
        'total_iam': 0.0
    }

    num_labels = confusion_matrix.shape[0]
    num_total_samples = np.sum(confusion_matrix).item()

    total_TP = 0

    for idx in range(num_labels):
        label_quotas = _compute_label_quotas(confusion_matrix, idx, num_total_samples)
        quotas['label_details'][idx] = label_quotas
        total_TP += label_quotas['TP']
        quotas['total_precision'] += label_quotas['precision']
        quotas['total_recall'] += label_quotas['recall']
        quotas['total_f1'] += label_quotas['f1']
        quotas['total_cba'] += label_quotas['cba']
        quotas['total_iam'] += label_quotas['iam']

    quotas['total_accuracy'] = total_TP / num_total_samples
    quotas['total_precision'] /= num_labels
    quotas['total_recall'] /= num_labels
    quotas['total_f1'] /= num_labels
    quotas['total_cba'] /= num_labels
    quotas['total_iam'] /= num_labels

    return quotas


# Plot confusion matrix and save it to designated path.
def plot_confusion_matrix(confusion_matrix: np.ndarray, path: str):
    '''
    :param confusion_matrix: np.ndarray, of shape (label_dim, label_dim), which is not normalized yet.
        confusion_matrix[i][j] represents count of those predictions which recognize label i as label j.
    :param path:
    :return:
    '''

    # Normalizing confusion matrix.
    row_sums = np.sum(confusion_matrix, axis=1).reshape((-1, 1))
    normalized_matrix = confusion_matrix / row_sums

    # Initializing figure.
    label_dim = confusion_matrix.shape[0]
    plt.figure(figsize=(int(label_dim * 0.6), int(label_dim * 0.5)))

    # Plot heap-map and save.
    cf_list = normalized_matrix.tolist()
    df_cf = pd.DataFrame(cf_list,
                         index=[str(i) for i in range(label_dim)],
                         columns=[str(i) for i in range(label_dim)])

    sn.heatmap(df_cf, annot=False, cmap='YlGnBu')
    plt.savefig(path)