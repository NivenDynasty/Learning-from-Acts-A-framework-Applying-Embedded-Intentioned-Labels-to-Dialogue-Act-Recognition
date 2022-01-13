import argparse

from custom_utils import *


def main(dataset_name, task_id):

    dataset_dir = 'Data/' + dataset_name + '/'
    figure_dir = 'Figures/'
    figure_path = figure_dir + dataset_name

    # Do sequence length analysis.
    if task_id == 1:
        analysis_seq_len(dataset_dir)

    # Do sequence token complexity analysis.
    elif task_id == 2:
        analysis_token_complexity(dataset_dir)

    # Do label frequency analysis.
    elif task_id == 3:
        analysis_label_freq(dataset_name, dataset_dir, figure_path + '_label_frequency.png')

    # Do label corresponding sequences' length analysis.
    elif task_id == 4:
        analysis_label_seq_len(dataset_dir)

    # Plot experimental results in a figure.
    elif task_id == 5:
        select_models = []
        continue_select = True

        while continue_select:
            print('Enter selected models, double enter to quit.\n>>>')
            input_model = str(input())
            if len(input_model) < 1:
                continue_select = False
            else:
                select_models.append(input_model)

        plot_experimental_results(dataset_name, dataset_dir, figure_path + '_experiment.png', select_models)


if __name__ == '__main__':

    print('Please input the name of dataset: ')
    dataset_name = str(input())

    print('Please input the task you want to do: ')
    print('1. Sequence length.')
    print('2. Sequence token complexity.')
    print('3. Label frequency.')
    print('4. Label corresponding sequence length')
    print('5. Visualize experimental results')
    print('>>> ', end='')
    task_id = int(input())

    main(dataset_name, task_id)