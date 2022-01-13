# Dialog Act Recognition (Classification) experiments implemented by Pytorch

My own experiments on dialog act recognition (DAR) tasks. Maybe helpful to others.

## Prerequisites

- **Python** 3.x, my version is 3.6.
- **Pytorch** 1.3.0 or higher, suggested version is 1.8.0+cuda, better with NVIDIA CUDA support, but CPU-Only supportive.
- **TorchText**, suggested version is 0.4.0.
- **Transformers**, suggested version is 3.3.1.

## Training

Datasets are prepared in folder 'Data/'. Each sub-folder in this directory corresponds to a unique dataset.

Default argument settings have been implemented in main.py. For a single trial, just run this command:
```bash
python main.py
```

## Arguments explanation

If you want to customize your own training, an easy way is just to customize those training arguments. For example, if you want to change the dataset, or model, you can type in command like this:
```bash
python main.py --dataset_name SwDA --model_name Bert
```

There are some other arguments that suggested to be given. In my own training, to make sure everything is under control. I usually type in command like this:
```bash
python main.py --cuda_idx 0 --dataset_name SwDA --model_name Bert --batch_size 32 --learning_rate 1e-4 --num_epochs 100
```

Here are explanations of arguments used:
- **cuda_idx(Recommended)**: Type int, the cuda index used for training. If you do not want to use cuda, you can set it to -1. If cuda is not available on your computer, whether cuda_idx is specifically set or not does not differ. **_Default is 0._**
- **dataset_name(Recommended)**: Type str, the dataset to train on. **_Default is 'MRDA'._**
- convert_to_csv: Type bool, whether to re-process original dataset files to .csv format. My program use 'train.csv', 'dev.csv' and 'test.csv' as data sources. If some of these .csv files are not found, my program will automatically re-generate them from original data files (train.txt, dev.txt, test.txt). **_Default is False_**.
- **model_name(Recommended)**: Type str, the model used on training, e.g. Linear, Bert, DistillBert, etc.
- use_label_embed: Type bool, whether to use label-embedding techniques. **_Default is False._**
- label_embed_ver: Type int, the version of label embedding technique. 1 represents logit-based embedding, 2 represents semantics-based embedding.
- save_model_dir: Type str, the folder to save models. **_Default is 'SavedModels/'._**
- save_model_path: Type str, the absolute path to save your model. If not given, my program will automatically generate one. **_Default is None._**
- load_saved_model: Type bool, whether to load saved model from save_model_path. This is useful on continuous training. **_Default is False._**
- freeze_pre_trained: Type bool, whether to freeze pre-trained model in fine-tuning. **_Default is False._**
- word2vec: Type str, pre-trained word embeddings to build vocab. This makes influence on experiment only when not using pre-trained modules such as Bert and XLNet. **_Default is 'glove.6B.300d'._**
- figure_dir: Type str, the folder to save experimental graphs. **_Default is 'Figures/'._**
- figure_path: Type str, the path to save confusion matrix figures. If not given, my program will automatically generate one, based on your model_name, dataset_name and the time you run this program. **_Default is None._**
- log_dir: Type str, the folder to save log files. **_Default is 'LogFiles/'._**
- log_path: Type str, the path to save log files. If not given, my program will automatically generate one, based on your model_name, dataset_name and the time you run this program. **_Default is None._**
- **max_seq_len**: Type int, max length of utterance sequences. **_Default is 25._**
- **hidden_dim**: Type int, the dimension of hidden states when using model containing hidden units. **_Default is 256._**
- **batch_size(Recommended)**: Type int, the size of a single batch. **_Default is 32._**
- **learning_rate(Recommended)**: Type float. This should be set properly due to different training performances of different models. **_Default is 1e-1._**
- weight_decay: Type float. **_Default is 5e-5._**
- **num_epochs(Recommended)**: Type int, the designated count of training epochs. Epochs of validating is the same as that of training. **_Default is 10._**
- loss_func: Type str, the loss function you use. In this version, only 'CrossEntropyLoss' will be used, but I will extend this in the future.
- optimizer: Type str, the optimizer you use. **_Default is 'SGD'._**
- do_train: Whether to do training. **_Default is True._**
- do_dev: Whether to do validating. This is of insignificant when do_train is set to False. **_Default is True._**
- do_test: Whether to do testing. **_Default is True._**

## Codes' structure.

```
Root
├── main.py
├── analysis.py
├── custom_utils.py
├── custom_models.py
└── custom_datasets.py
```
- **main.py**: Main program to run.
- **analysis.py**: Just for analyzing datasets, as well as ... experimental results.
- **custom_utils.py**: Containing useful utils, e.g. data pre-processing, legality checking.
- **custom_models.py**: Containing all models that are used in my experiment. Some of the models are not fully implemented yet (e.g. HLSN).
- **custom_datasets.py**: Customized datasets, but I did not use it in later experiments.

## Customized Dataset

To train on your own dataset, here are the steps you need to follow:
1. Put your dataset folder in 'Data/'. In your own dataset folder, either of these files should be contained:
    - train.txt, dev.txt, test.txt OR
    - train, dev, test
2. Open code **_'custom_utils.py'_**, and add the name of your dataset in list **_VALID_DATASET**.
3. In **_'custom_utils.py'_**, find function **judge_output_dim_dataset()**, and manually set the output dimension (label dimension) of your dataset (just as the way I set them up before).