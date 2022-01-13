from transformers import AutoTokenizer

from torchtext.data import Field, BucketIterator, Iterator, TabularDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD, Adam
import torch.nn as nn
import torch

import os
import sys
import copy
import time
import shutil
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import customized utils.
from custom_utils import *
from custom_models import *
from custom_datasets import *


# Main.
def main(args):

    # Recording Time.
    # Time information will be recorded in logging files and output figures.
    program_start_time = time.time()
    dt = datetime.datetime.now()
    dt_list = [dt.month, dt.day, dt.hour, dt.minute]

    # Index of cuda.
    # If you don't have cuda on your machine, cuda_idx will be automatically set to -1.
    # If cuda_idx is less than 0, your machine will use CPU for training, which is not recommended in large-scale model.
    cuda_idx = args.cuda_idx if torch.cuda.is_available() else -1
    device = torch.device('cuda:{}'.format(cuda_idx)) if cuda_idx >= 0 else torch.device('cpu')

    # Decide which dataset to train on.
    # Dataset's name will affect some of training arguments.
    #
    #   1. Dataset checking.
    #       Firstly, make sure your dataset folder is put under './Data/', and the folder should be named as dataset.
    #       Secondly, go to custom_utils.py, and add your LOWER dataset name in _VALID_DATASET list.
    #       Thirdly, still in custom_utils.py, add the output dimension of your dataset in _OUTPUT_DIM list.
    #
    #   2. Creating saved model path's name from dataset's name, and so on. This is automatically done by the program.
    #       Designated dataset folder has the same name as dataset_name, which is put under './Data/'.
    #       Saved model's name, logging file's name and output figure's name will created depending on dataset_name.
    #
    dataset_name = args.dataset_name
    check_valid_dataset(dataset_name)
    output_dim = judge_output_dim_dataset(dataset_name)
    data_dir = 'Data/' + dataset_name + '/'

    # Judge whether it is essential to convert original data file to format .csv file.
    convert_to_csv = args.convert_to_csv
    convert_dataset_to_csv(data_dir, output_dim, convert_to_csv, -1)

    # Get calculated percentages of labels. This is implemented in custom_utils.py.
    label_percentage_dict = get_label_percentages()
    label_weights = [1.0 - label_percentage_dict[k] for k in label_percentage_dict.keys()]

    # Select models, either pre-trained-model contained.
    # Model name is valid when its lower form is one of those implemented in custom_models.py.
    # For example, type in 'Bert', 'BERT', 'bert' will all build up your model on Bert implemented in custom_models.py.
    # My program will automatically judge whether there is a pre-trained module in model.
    model_name = args.model_name
    pre_trained_model = pre_trained_inside(model_name)

    # Tensorboard graph path.
    tensorboard_graph_path = 'Runs/' + model_name
    if os.path.exists(tensorboard_graph_path):
        shutil.rmtree(tensorboard_graph_path)
    model_writer = SummaryWriter(tensorboard_graph_path)

    # Whether to use label embedding-techniques.
    # If this is set to true, the training procedure will be somewhat different.
    use_label_embed = args.use_label_embed
    label_embed_ver = args.label_embed_ver
    convert_label_embed_version(label_embed_ver)

    # The path of model saving. If None given, the program automatically create a path by the name of model.
    # It is recommended not to explicitly give the path of saved model.
    save_model_dir = args.save_model_dir
    save_model_path = args.save_model_path
    if save_model_path is None:
        if use_label_embed:
            save_model_path = model_name + '(LE' + str(label_embed_ver) + ')_' + dataset_name + '_' + args.loss_func + '_' + args.optimizer + '.pt'
        else:
            save_model_path = model_name + '_' + dataset_name + '_' + args.loss_func + '_' + args.optimizer + '.pt'
    if save_model_dir is not None:
        save_model_path = save_model_dir + save_model_path

    # Deal with word2vec, and set the embed_dim automatically by given vectors.
    # Recommend to use GloVe as word2vec embedding. For example, 'glove.6B.300d'.
    word2vec_name = args.word2vec
    check_valid_word2vec(word2vec_name)
    embed_dim = judge_dim_word2vec(word2vec_name)

    # The folder that contains figures. There are confusion matrix and losses that need to be plotted.
    figure_dir = args.figure_dir
    figure_path = model_name + dataset_name + '_' + '_'.join(str(t) for t in dt_list) + '.png' if args.figure_path is None \
        else args.figure_path

    # The folder that contains logs.
    log_dir = args.log_dir
    log_path = model_name + dataset_name + '_' + '_'.join(str(t) for t in dt_list) + '.txt' if args.log_path is None \
        else args.log_path

    # Restrict max length of each sequence.
    # In dialog act tasks, average length of sequence is shorter than that in other traditional tasks.\
    # It's suggested to set max_seq_len in range of [15, 30].
    max_seq_len = args.max_seq_len

    # Loading model's hyper-parameters to parameters' dict.
    param_dict = {}
    param_dict['device'] = device
    param_dict['use_label_embed'] = use_label_embed
    param_dict['label_embed_ver'] = label_embed_ver
    param_dict['pre_trained_model'] = pre_trained_model
    param_dict['freeze_pre_trained'] = args.freeze_pre_trained
    param_dict['pre_trained_cache'] = 'model_cache/'
    param_dict['max_seq_len'] = max_seq_len
    param_dict['embed_dim'] = embed_dim
    param_dict['hidden_dim'] = args.hidden_dim
    param_dict['output_dim'] = output_dim
    param_dict['num_heads'] = args.num_heads

    # The path of figures' / logs' writing.
    lf = open(log_dir + log_path, 'w', encoding='utf-8', newline='\n')

    # Log arguments.
    all_arguments = vars(args)
    lf.write('*********************************** Arguments overview: ***********************************\n')
    for k, v in sorted(all_arguments.items()):
        lf.write('>>> Arg {}: {}\n'.format(k, v))
    lf.write('\n\n')

    # Other user-defined training parameters.
    load_saved_model = args.load_saved_model if args.do_train else True
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs if args.do_train else 0
    dev_per_epochs = args.dev_per_epochs

    do_train = args.do_train
    do_dev = args.do_dev
    do_test = args.do_test

    # Fields
    text_field = Field(
        tokenize='basic_english',
        lower=True,
        batch_first=True,
        fix_length=max_seq_len,
    )
    label_field = Field(
        sequential=False,
        use_vocab=False,
        batch_first=True,
        dtype=torch.long
    )

    # When model contains pre-trained module inside, the way to build data-fields would be different.
    pad_index = 1
    unk_index = 0

    # When using pre-trained model, padding index and unknown index should be defined by the model itself.
    if pre_trained_model is not None:

        # Get tokenizer from pre-trained model.
        tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
        pad_index = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        unk_index = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

        # Update fields.
        text_field = Field(tokenize=tokenizer.encode,
                           lower=False,
                           batch_first=True,
                           fix_length=max_seq_len,
                           include_lengths=False,
                           use_vocab=False,
                           pad_token=pad_index,
                           unk_token=unk_index)

        # Transmit pad index to custom_models.py
        transmit_pad_index(pad_index)

    # When model is not using pre-trained module, build vectors from pre-trained embeddings.
    else:

        # Read data-frame from .csv file.
        data_frame = {}

        for dataset_type in ['train', 'dev', 'test']:
            data_frame[dataset_type] = pd.read_csv(data_dir + dataset_type + '.csv')
            ltoi = {l: i for i, l in enumerate(data_frame[dataset_type]['label'].unique())}
            data_frame[dataset_type]['label'] = data_frame[dataset_type]['label'].apply(lambda y: ltoi[y])

        # Build vocabulary from train data.
        processed_text = data_frame['train']['text'].apply(lambda x: text_field.preprocess(x))
        text_field.build_vocab(processed_text, vectors=word2vec_name)

        # Get the vocab instance.
        vocab = text_field.vocab
        print('Not using pre-trained tokenizer.')
        print('Successfully build pre-trained embedding from {}.'.format(word2vec_name))
        print('Vocab vectors\' size: {}.'.format(vocab.vectors.size()))
        param_dict['vocab_size'] = len(vocab)
        param_dict['vocab_embedding'] = vocab.vectors

        # Transmit pad index to custom_models.py
        transmit_pad_index(1)

    # Loading tabular-dataset.
    train, dev, test = TabularDataset.splits(path=data_dir,
                                             train='train.csv',
                                             validation='dev.csv',
                                             test='test.csv',
                                             fields=[('label', label_field), ('text', text_field)],
                                             format='CSV',
                                             skip_header=True)

    '''
        The beginning of data-building, model-loading and training.
    '''

    # Get Bucket-iterators.
    train_iter = BucketIterator(train, batch_size=batch_size, shuffle=False, sort=False, device=device, train=True)
    dev_iter = BucketIterator(dev, batch_size=batch_size, shuffle=False, sort=False, device=device, train=False)
    test_iter = Iterator(test, batch_size=batch_size, shuffle=False, sort=False, device=device, train=False)

    len_train_iter = len(train_iter)
    len_dev_iter = len(dev_iter)

    # Initialize ModelParam instance and neural network.
    model_param = ModelParam(param_dict)
    model = get_model(model_name, model_param).to(device)
    print('Using model {}, pre-trained module is {}.'.format(model_name, pre_trained_model))
    if load_saved_model and os.path.exists(save_model_path):
        model = torch.load(save_model_path, map_location=device)
        print('Loading saved model from \'{}\''.format(save_model_path))

    # Add graph to tensorboard directory before formally training.
    # This part is not necessary. Annotate it if model-visualizing is not needed!
    '''
    for batch in train_iter:
        x = batch.text.type(torch.LongTensor).to(device)
        labels = batch.label.to(device)
        masks = batch.text.not_equal(pad_index).type(torch.FloatTensor).to(device)
        try:
            model_writer.add_graph(model, [x, labels, masks])
            model_writer.close()
        except RuntimeError:
            print("Runtime Error occurred when trying to add graph to tensorboard writer.")
            model_writer.close()
        break
    '''

    best_dev_model = get_model(model_name, model_param).to(device)

    '''
    if model is not None:
        for param in model.named_parameters():
            print('>>> Param [{}]\'s size is {}.'.format(param[0], param[1].size()))
            lf.write('Model param [{}] is of size [{}]\n'.format(param[0], param[1].size()))
    '''

    # Loss function. Use CrossEntropyLoss() by default.
    loss_func = nn.CrossEntropyLoss()
    loss_func_name = 'Cross Entropy Loss'
    if use_label_embed:
        if label_embed_ver == 1:
            loss_func = LabelEmbeddingLoss()
            loss_func_name = 'Label Embedding Loss'
        else:
            loss_func = ContextLabelEmbedLoss()
            loss_func_name = 'Context Embed Loss'
    elif args.loss_func.lower() == 'weighted':
        model_param.build_loss_weights(label_weights, False)
        loss_func = WeightedLoss(model_param)
        loss_func_name = 'Weighted Loss'

    print('Loss function used: {}.'.format(loss_func_name))
    lf.write('Loss function used: {}.\n'.format(loss_func_name))

    # Optimizer. Use SGD by default.
    optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_name = 'SGD'
    if args.optimizer.lower() == 'adam':
        optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_name = 'Adam'
    print('Optimizer used: {}.'.format(optimizer_name))
    lf.write('Optimizer used: {}.\n'.format(optimizer_name))

    min_dev_loss = 99999.0
    max_dev_acc = 0.0
    min_loss_epoch = 0

    train_epochs = []
    dev_epochs = []
    train_losses = []
    dev_losses = []

    if do_train:

        for epoch in range(num_epochs):

            # Read signal file.
            # This is useful when you want to end up training when not reaching the maximum epoch in advance.
            # You can change the signal in SIGNAL to realize this.
            if os.path.exists('./SIGNAL.txt'):
                sf = open('./SIGNAL.txt', 'r', encoding='utf-8')

                try:
                    if sf.readlines()[0].rstrip() == 'STOP':
                        print('Training ended up in advance in epoch {}, because of STOP signal in SIGNAL.txt file.'.format(
                            epoch + 1
                        ))
                        lf.write('Training ended up in advance in epoch {}, because of STOP signal in SIGNAL.txt file.\n'.format(
                            epoch + 1
                        ))
                        break
                except IndexError:
                    pass

                sf.close()

            train_epochs.append(epoch + 1)

            epoch_start_time = time.time()
            train_batch_idx = 0

            # Start training.
            total_train_loss = 0.0
            for batch in train_iter:

                train_batch_idx += 1

                # Abandon those samples in wrong-sized batch.
                if not (batch.text.size()[1] == max_seq_len and batch.text.size()[0] == batch_size):
                    continue

                batch_start_time = time.time()

                optimizer.zero_grad()

                # Extract data.
                x = batch.text.type(torch.LongTensor).to(device)
                labels = batch.label.to(device)
                masks = batch.text.not_equal(pad_index).type(torch.FloatTensor).to(device)

                # The output of model. Composition of output could be complex.
                outputs = model(x, labels, masks)
                loss = loss_func(outputs, labels)

                loss.backward()
                optimizer.step()

                # Record total loss.
                total_train_loss += loss.item()

                batch_time = time.time() - batch_start_time

                # Log info.
                sys.stdout.write('\rTrain epoch {}/{}, batch {}/{} | Loss {:.3f} | Time left: {:.2f} sec'.format(
                    epoch + 1, num_epochs, train_batch_idx, len_train_iter, loss.item() / batch_size,
                    (batch_time + 1e-4) * 1.75 * (len_train_iter - train_batch_idx)
                ))

            dev_batch_idx = 0
            train_losses.append(total_train_loss / len_train_iter)

            # Start validating.
            if do_dev and (epoch + 1) % dev_per_epochs == 0:

                sys.stdout.write('\r                                                                                  ')
                lf.write('******************** Validating epoch {} ********************\n'.format(epoch + 1))

                total_dev_loss = 0.0
                total_samples = 1
                right_samples = 1
                
                for batch in dev_iter:

                    dev_batch_idx += 1

                    # Abandon those samples in wrong-sized batch.
                    if not batch.text.size()[0] == batch_size:
                        continue

                    with torch.no_grad():

                        total_samples += batch_size
                        optimizer.zero_grad()

                        # Extract data.
                        x = batch.text.type(torch.LongTensor).to(device)
                        labels = batch.label.to(device)
                        masks = batch.text.not_equal(pad_index).type(torch.FloatTensor).to(device)

                        # The prediction of model, and its corresponding loss.
                        outputs = model(x, labels, masks)
                        predictions = torch.argmax(outputs[0], dim=1) if type(outputs) is tuple \
                            else torch.argmax(outputs, dim=1)
                        right_samples += sum(predictions == labels).item()

                        loss = loss_func(outputs, labels)
                        total_dev_loss += loss.item() / batch_size

                        sys.stdout.write('\rValidate batch {}/{} | Loss {:.4f}'.format(
                            dev_batch_idx, len_dev_iter, loss.item() / batch_size)
                        )
                        lf.write('>>> Validating batch {}/{}, average loss = {:.6f}\n'.format(
                            dev_batch_idx, len_dev_iter, loss.item() / batch_size)
                        )

                cur_acc = 1.0 * right_samples / total_samples
                cur_loss = total_dev_loss / len_dev_iter * batch_size

                # Record loss and accuracy.
                if (max_dev_acc < cur_acc - 1e-3) or (min_dev_loss > cur_loss + 5e-3):

                    if min_dev_loss > cur_loss:
                        min_dev_loss = cur_loss

                    if max_dev_acc < cur_acc:
                        max_dev_acc = cur_acc

                    min_loss_epoch = epoch + 1
                    best_dev_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    torch.save(model, save_model_path)

                epoch_time = time.time() - epoch_start_time

                sys.stdout.write('\rEpoch {}/{} val acc: [{:.3f}] ({} in {}), time spent {:.3f} sec. Val loss {:.6f}. Best val epoch is {}, best val acc [{:.3f}].\n'.format(
                    epoch + 1, num_epochs, right_samples / total_samples, right_samples, total_samples, epoch_time, total_dev_loss / len_dev_iter, min_loss_epoch, max_dev_acc)
                )
                lf.write('End of epoch {}, time spent {:.3f} seconds, val loss {:.2f}, val acc {:.2f}, best epoch {}.\n\n'.format(
                    epoch + 1,  epoch_time, total_dev_loss / len_dev_iter, right_samples / total_samples, min_loss_epoch)
                )
                dev_epochs.append(epoch + 1)
                dev_losses.append(total_dev_loss / len_dev_iter)

    # Plot losses after training and validating.
    if do_train:

        plt.plot(train_epochs, train_losses, label='train losses')
        plt.plot(dev_epochs, dev_losses, label='dev losses')
        plt.legend()
        plt.savefig(figure_dir + 'Loss_' + figure_path, bbox_inches='tight')

    confusion_matrix = np.zeros((param_dict['output_dim'], param_dict['output_dim']), dtype=np.int)

    # Testing.
    if do_test:

        if not do_train:
            best_dev_model = torch.load(save_model_path, map_location=device)

        best_dev_model.eval()

        total_samples = 1
        right_samples = 1
        for batch in test_iter:

            # Abandon those samples in wrong-sized batch.
            if not batch.text.size()[0] == batch_size:
                continue

            with torch.no_grad():
                total_samples += batch_size
                optimizer.zero_grad()

                # Extract data.
                x = batch.text.type(torch.LongTensor).to(device)
                labels = batch.label.to(device)
                masks = batch.text.not_equal(pad_index).type(torch.FloatTensor).to(device)

                # The prediction of model, and its corresponding accuracy.
                outputs = best_dev_model(x, labels, masks)
                predictions = torch.argmax(outputs[0], dim=1) if type(outputs) is tuple \
                    else torch.argmax(outputs, dim=1)
                right_samples += sum(predictions == labels).item()

                # Fill confusion matrix.
                for idx in range(batch_size):
                    confusion_matrix[batch.label[idx].item()][predictions[idx].item()] += 1

        quotas = get_matrix_quotas(confusion_matrix)
        plot_confusion_matrix(confusion_matrix, figure_dir + 'CM_' + figure_path)

        program_time = time.time() - program_start_time

        lf.write('\nFinal reporting\n')
        print('\n\n*********************************** Reporting: ***********************************')
        print('>>> Total time spent: {} hour, {} minute.'.format(
            int(program_time / 3600),
            int(program_time - 3600 * int(program_time // 3600)))
        )
        print('>>> Training start at time slot {}.'.format(':'.join([str(d) for d in dt_list])))
        print('>>> Best validate epoch stopped at {} ({} in total).'.format(min_loss_epoch, num_epochs))
        print('>>> Total accuracy: [{:.3f}]'.format(quotas['total_accuracy']))
        print('>>> Total precision: [{:.3f}]\tTotal recall: [{:.3f}]\tTotal f1-score: [{:.3f}]'.format(
            quotas['total_precision'],
            quotas['total_recall'],
            quotas['total_f1'],
        ))
        print('>> Total cba: [{:.3f}]\tTotal iam: [{:.3f}]'.format(
            quotas['total_cba'],
            quotas['total_iam']
        ))

        lf.write('Time spent: {} hours, {} minutes.'.format(
            int(program_time / 3600),
            int(program_time - 3600 * int(program_time / 3600)) // 60)
        )
        lf.write('>>> Acc {:.3f}\n>>> Precision {:.3f}\n>>> Recall {:.3f}\n>>> F1 {:.3f}\n>>> CBA {:.3f}\n>>> IAM {:.3f}\n'.format(
            quotas['total_accuracy'],
            quotas['total_precision'],
            quotas['total_recall'],
            quotas['total_f1'],
            quotas['total_cba'],
            quotas['total_iam']
        ))

        lf.write('\n******************** Label reporting ********************\n')
        print('*********************************** Label details: ***********************************')
        for i in range(param_dict['output_dim']):
            print('> Label {}:\t count {}, accuracy {:.2f}, precision {:.2f}, recall {:.2f}, f1 {:.2f}, most predicted to be {}.'.format(
                i,
                quotas['label_details'][i]['count'],
                quotas['label_details'][i]['accuracy'],
                quotas['label_details'][i]['precision'],
                quotas['label_details'][i]['recall'],
                quotas['label_details'][i]['f1'],
                quotas['label_details'][i]['max_predicted']
            ))
            lf.write('>>> Label[{}](Total count: {})\n'.format(
                i,
                quotas['label_details'][i]['count'],
            ))
            lf.write('\t>>> Acc {:.2f}, TP {}, FP {}, FN {}, precision {:.2f}, recall {:.2f}, f1 {:.2f}, most predicted to be label[{}]\n'.format(
                quotas['label_details'][i]['accuracy'],
                quotas['label_details'][i]['TP'],
                quotas['label_details'][i]['FP'],
                quotas['label_details'][i]['FN'],
                quotas['label_details'][i]['precision'],
                quotas['label_details'][i]['recall'],
                quotas['label_details'][i]['f1'],
                quotas['label_details'][i]['max_predicted']
            ))

    lf.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    '''
        Simplest command is suggested to be like this:

        >>> python main.py --model_name [your model name] --learning_rate [your learning rate]

        For example, just type in:

        >>> python main.py --model_name Linear --learning_rate 1e-3

        If this is the first time to run on specific dataset, use --convert_to_csv.

        >>> python main.py --dataset_name SwDA --convert_to_csv True --model_name Linear --learning_rate 1e-3
    '''

    # Which cuda to use. If not using cuda, then set it to -1.
    parser.add_argument('--cuda_idx', type=int, required=False, default=0)

    # Current tmux window index, this is ONLY used for recording (IGNORE it when not running the codes on shell).
    parser.add_argument('--tmux_idx', type=int, required=False, default=-1)

    # Which dataset to use.
    parser.add_argument('--dataset_name', type=str, required=False, default='SwDA')

    # Is it required to convert original dataset to .csv file.
    parser.add_argument('--convert_to_csv', type=bool, required=False, default=False)

    # Which model to use.
    '''
        Available models:
        'Linear', 'DoubleLinear' ('MLP'), 'GRU', 'TextCNN', 
        'BERT', 'BERTRNN', 'DistillBERT', 'XLNet', 
        'Seq2Seq', 'HLSTM', 'HLSN'
    '''
    parser.add_argument('--model_name', type=str, required=False, default='WinSeq2SeqAttn')

    # Whether to use label-embedding technique.
    parser.add_argument('--use_label_embed', type=bool, required=False, default=False)

    # The version of label-embedding.
    parser.add_argument('--label_embed_ver', type=int, required=False, default=1)

    # Saving model inside a designated folder. This is not required when you are using absolute path of saving models.
    parser.add_argument('--save_model_dir', type=str, required=False, default='SavedModels/')

    # Saving model to designated path. If not given, this code will automatically create a proper one.
    parser.add_argument('--save_model_path', type=str, required=False, default=None)

    # Whether to load saved model from designated path.
    parser.add_argument('--load_saved_model', type=bool, required=False, default=True)

    # Whether to freeze weights in pre-trained models (If used). We highly recommend this to be set False.
    parser.add_argument('--freeze_pre_trained', type=bool, required=False, default=False)

    # Which pre-trained word2vec to use.
    '''
        Available word2vec choices:
        'charngram.100d', 'fasttext.en.300d', 'fasttext.simple.300d', 'glove.42B.300d', 'glove.840B.300d',
        'glove.twitter.27B.25d', 'glove.twitter.27B.50d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d',
        'glove.6B.50d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d'
    '''
    parser.add_argument('--word2vec', type=str, required=False, default='glove.6B.300d')

    # Figure folder.
    parser.add_argument('--figure_dir', type=str, required=False, default='Figures/')

    # Figure path to write. If not given, the program will automatically generate one.
    parser.add_argument('--figure_path', type=str, required=False, default=None)

    # Which folder to write log files in.
    parser.add_argument('--log_dir', type=str, required=False, default='LogFiles/')

    # Log path to write. If not given, the program will automatically generate one.
    parser.add_argument('--log_path', type=str, required=False, default=None)

    # The maximum of single sequence fed into network.
    parser.add_argument('--max_seq_len', type=int, required=False, default=25)

    # Hidden dim when using structures with hidden states.
    parser.add_argument('--hidden_dim', type=int, required=False, default=256)

    # The amount of heads when using multi-head attention module.
    parser.add_argument('--num_heads', type=int, required=False, default=2)

    # Size of training batch.
    parser.add_argument('--batch_size', type=int, required=False, default=32)

    # Learning rate.
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-1)

    # Learning weight decay.
    parser.add_argument('--weight_decay', type=float, required=False, default=5e-5)

    # Expected count of training epoch.
    parser.add_argument('--num_epochs', type=int, required=False, default=10)

    # Do validation per designated epochs.
    parser.add_argument('--dev_per_epochs', type=int, required=False, default=1)

    # The name of loss function to use.
    parser.add_argument('--loss_func', type=str, required=False, default='CrossEntropy')

    # The name of optimizer to use.
    parser.add_argument('--optimizer', type=str, required=False, default='SGD')

    # Whether to do train, do valid or do test.
    parser.add_argument('--do_train', type=bool, required=False, default=True)
    parser.add_argument('--do_dev', type=bool, required=False, default=True)
    parser.add_argument('--do_test', type=bool, required=False, default=True)

    args = parser.parse_args()
    main(args)