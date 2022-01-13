import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, DistilBertModel, XLNetModel


# This global variables.
_PAD_INDEX = 0
_LABEL_EMBED_VERSION = 1


# Convert version of label-embedding from main.py.
def convert_label_embed_version(version: int):
    global _LABEL_EMBED_VERSION
    _LABEL_EMBED_VERSION = version


# A class that contains useful hyper-parameters of model.
# Use the instance of class ModelParam to initialize all different models.
class ModelParam(object):

    def __init__(self, param_dict: dict):

        self.device = param_dict.get('device', torch.device('cpu'))
        self.vocab_size = param_dict.get('vocab_size', 0)
        self.vocab_embedding = param_dict.get('vocab_embedding', None)
        self.use_label_embed = param_dict.get('use_label_embed', False)
        self.label_embed_ver = param_dict.get('label_embed_ver', 1)
        self.pre_trained_model = param_dict.get('pre_trained_model', None)
        self.pre_trained_cache = param_dict.get('pre_trained_cache', 'model_cache/')
        self.freeze_pre_trained = param_dict.get('freeze_pre_trained', False)
        self.max_seq_len = param_dict.get('max_seq_len', 0)
        self.num_heads = param_dict.get('num_heads', 0)
        self.embed_dim = param_dict.get('embed_dim', 0)
        self.hidden_dim = param_dict.get('hidden_dim', 0)
        self.output_dim = param_dict.get('output_dim', 0)

        # Building loss weights, of shape [output_dim], init randomly by default.
        self.loss_weights = torch.randn((self.output_dim), requires_grad=False).to(self.device)
        nn.init.uniform_(self.loss_weights, 0.0, 1.0)

    def build_loss_weights(self, weights: list, do_soft_max: bool):

        assert len(weights) == self.loss_weights.size(0), \
            'Given loss function weight list is in length {}， which mismatch the label dim of your dataset: {}'.format(
                len(weights), self.loss_weights.size(0)
            )

        weights_tensor = torch.FloatTensor(weights)
        weights_tensor.requires_grad = False
        if do_soft_max:
            weights_tensor = F.softmax(weights_tensor)
        self.loss_weights.data.copy_(weights_tensor)


# Automatically initialize a series of neural network, and return it.
def get_model(model_name: str, model_param: ModelParam):

    lower_model_name = model_name.lower()
    if lower_model_name == 'linear':
        return Linear(model_param)
    elif lower_model_name in ['doublelinear', 'mlp']:
        return DoubleLinear(model_param)
    elif lower_model_name in ['cnn', 'textcnn']:
        return TextCNN(model_param)
    elif lower_model_name == 'gru':
        return GRU(model_param)
    elif lower_model_name == 'bert':
        return Bert(model_param)
    elif lower_model_name == 'bertrnn':
        return BertRNN(model_param)
    elif lower_model_name in ['distil', 'distill', 'distilled', 'distilbert', 'distillbert', 'distilledbert']:
        return DistillBert(model_param)
    elif lower_model_name == 'xlnet':
        return XlNet(model_param)
    elif lower_model_name == 'hlstm':
        return HLstm(model_param)
    elif 'seq2seq' in lower_model_name:
        return WinSeq2SeqAttn(model_param)
    else:
        raise AssertionError('Given model name [{}] is invalid.'.format(model_name))


# Check whether given model contains a pre-trained module inside.
def pre_trained_inside(model_name: str):

    lower_model_name = model_name.lower()
    if lower_model_name in ['bert']:
        return 'bert-base-uncased'
    elif lower_model_name in ['distil', 'distill', 'distilled', 'distilbert', 'distillbert', 'distilledbert']:
        return 'distilbert-base-uncased'
    elif lower_model_name in ['xlnet']:
        return 'xlnet-base-cased'
    else:
        return None


# Transmit padding index from main.py
def transmit_pad_index(new_pad_index: int):

    global _PAD_INDEX
    _PAD_INDEX = new_pad_index


# Customized loss function.
# This is not implemented systematically yet.
class WeightedLoss(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(WeightedLoss, self).__init__()
        self.device = model_param.device

        # torch.FloatTensor of shape [output_dim].
        self.loss_weights = model_param.loss_weights

    def forward(self, inputs, targets):
        '''
        :param inputs: torch.FloatTensor, of shape [batch_size, output_dim].
        :param targets: torch.LongTensor, of shape [batch_size].
        :return:
        '''

        # Get hyper-parameters.
        batch_size = inputs.size(0)
        output_dim = inputs.size(1)

        # Build one-hot vectors.
        trg_onehot = torch.randn((batch_size, output_dim)).to(self.device)
        trg_onehot.zero_()
        trg_onehot.scatter_(1, targets.unsqueeze(1), 1)

        # Calculate weighted loss.
        return torch.mean(self.loss_weights * ((inputs - trg_onehot) ** 2))


# Probability form of Cross-Entropy Loss.
# This is used when label embedding module is activated.
class ProbCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(ProbCrossEntropyLoss, self).__init__()

    def forward(self, input, targets):
        '''
        :param input: Predicted logits, torch.FloatTensor, of shape [batch_size, output_dim].
        :param targets: The probability form of labels, torch.FloatTensor, of shape [batch_size, output_dim].
        :return:
        '''

        # [batch_size, output_dim].
        soft_input = F.log_softmax(input, dim=-1)

        # [batch_size].
        loss = torch.sum(targets * soft_input, 1)
        return loss


# Loss calculated when label embedding module is activated.
class LabelEmbeddingLoss(nn.Module):

    def __init__(self):
        super(LabelEmbeddingLoss, self).__init__()

        self.alpha = 0.9
        self.beta = 0.5
        self.tau = 2
        self.prob_loss = ProbCrossEntropyLoss()

    def forward(self, inputs, targets):
        '''
        :param inputs: Tuple, consist of 3 parts:
            out1: Original prediction of model, torch.FloatTensor, [batch_size, output_dim].
            out2: Output prediction from which does not require gradient, torch.FloatTensor, [batch_size, output_dim].
            label_embeds: Label embedding results, torch.FloatTensor, [batch_size, output_dim].
        :param targets: Original true label, torch.LongTensor, [batch_size].
        :return final_loss: Total loss computed from some modules.
        '''

        # Disassemble inputs.
        out1, out2, label_embeds = inputs

        out2_soft = F.softmax(out2, dim=-1)
        tau2_soft = F.softmax(out2 / self.tau, dim=-1).detach()
        embed_soft = F.softmax(label_embeds, dim=-1).detach()

        # The loss computed from predicted logits and true labels.
        loss1_pred = F.cross_entropy(out1, targets)

        # The loss computed from predicted logits and label embeddings.
        loss1_embed = -torch.mean(self.prob_loss(out1, embed_soft))

        # The loss computed from none-gradient connected outputs and true labels.
        loss2_pred = F.cross_entropy(out2, targets)

        # Predicted logits from out2.
        _, pred = torch.max(out2, 1)

        # Only those correct predictions will enable loss-backward in loss2_embed.
        mask = pred.eq(targets).float().detach()
        loss2_embed = -torch.sum(self.prob_loss(label_embeds, tau2_soft) * mask) / (torch.sum(mask) + 1e-9)

        # Regular operation.
        # [batch_size, 1].
        gap = torch.gather(out2_soft, 1, targets.view(-1, 1)) - self.alpha
        loss_regular = torch.sum(F.relu(gap))

        # Compute final loss.
        final_loss = self.beta * loss1_pred + (1 - self.beta) * loss1_embed + loss2_pred + loss2_embed + loss_regular

        return final_loss


# Private module: Label embedding network.
# This is used for testing whether label embedding techniques does help in classification.
class LabelEmbedding(nn.Module):

    def __init__(self, linear_dim: int, output_dim: int, device=torch.device('cpu')):
        '''
        :param linear_dim: The input dimension of the last fc, in other models.
        :param output_dim: The output dimension, equal to number of category of labels.
        :param device: The device used for training.
        '''
        super(LabelEmbedding, self).__init__()

        self.Linear = nn.Linear(linear_dim, output_dim)
        self.Embedding = nn.Embedding(output_dim, output_dim)
        self.Embedding.weight = nn.Parameter(torch.eye(output_dim))

    def get_embedding(self):

        return self.Embedding.weight

    def forward(self, features, labels):
        '''
        :param features: torch.FloatTensor, of shape [batch_size, linear_dim].
        :param labels: torch.LongTensor, of shape [batch_size].
        :return out_features: None-gradient form of prediction, torch.FloatTensor, of shape [batch_size, output_dim].
        :return out_embeddings: The embedding result of labels, torch.FloatTensor, of shape [batch_size, output_dim].
        :return weight: Weight data of Embedding, torch.FloatTensor, of shape [output_dim, output_dim].
        '''

        global _LABEL_EMBED_VERSION

        # [batch_size, output_dim].
        out_features = self.Linear(features.detach())

        # [batch_size, output_dim].
        out_embeddings = self.Embedding(labels)

        return out_features, out_embeddings, self.Embedding.weight


# List-output form of Cosine-Embedding Loss.
class ListCosineEmbeddingLoss(nn.Module):

    def __init__(self):
        super(ListCosineEmbeddingLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, features, embeddings):
        '''
        :param features: [batch_size, embed_dim].
        :param embeddings: [batch_size, embed_dim].
        :return lost_list: torch.FloatTensor, of shape [batch_size], each value is ranging from 0~2.
        '''

        batch_size = features.size(0)
        device = features.device
        oner = torch.ones((batch_size), dtype=torch.float).to(device)

        return oner - self.cos_sim(features, embeddings)


# Loss calculated when context label embedding module is activated.
class ContextLabelEmbedLoss(nn.Module):

    def __init__(self):
        super(ContextLabelEmbedLoss, self).__init__()

        self.cos_loss = ListCosineEmbeddingLoss()
        self.nn_cos_loss = nn.CosineEmbeddingLoss()
        self.beta = 0.5

    def forward(self, inputs, targets):
        '''
        :param inputs: Tuple, consist of 4 parts:
            logits: Original prediction, torch.FloatTensor, of shape [batch_size, output_dim].
            out_features: Fully-connected context features. This is learnt in ContextLabelEmbed model, [batch_size, embed_dim].
            out_logits: Prediction from label embed module, of shape [batch_size, output_dim].
            out_embeddings: Label embedding results, [batch_size, embed_dim].
        :param targets: Original true label, torch.LongTensor, of shape [batch_size].
        :return final_loss: Total loss computed.
        '''
        logits, out_features, out_logits, out_embeddings = inputs

        # Get batch size and embed dim.
        batch_size = out_features.size(0)
        embed_dim = out_features.size(1)

        # The loss computed from predicted logits and true labels.
        loss_pred1 = F.cross_entropy(logits, targets)
        _, pred1 = torch.max(logits, 1)

        '''
        # The loss computed from embedding logits and true labels.
        loss_pred2 = F.cross_entropy(out_logits, targets)
        _, pred2 = torch.max(out_logits, 1)
        '''
        soft_features = F.softmax(out_features, dim=-1)
        soft_embeddings = F.softmax(out_embeddings, dim=-1)

        # Only those correct predictions will enable loss-backward.
        # [batch_size].
        pred_mask = pred1.eq(targets).float().detach()

        # The loss computed from context features and embedded-label features.
        # Range: 0~2.
        loss_cos = torch.sum(self.cos_loss(soft_features, soft_embeddings) * pred_mask) / (torch.sum(pred_mask) + 1e-6)
        # loss_cos = self.nn_cos_loss(out_features, out_embeddings, torch.ones((batch_size), dtype=torch.long).to(logits.device))

        return self.beta * loss_pred1 + (1 - self.beta) * loss_cos


# Private module: Context-based Label embedding network.
class ContextLabelEmbed(nn.Module):

    def __init__(self, embed_dim: int, output_dim: int, device = torch.device('cpu')):
        super(ContextLabelEmbed, self).__init__()

        # Acquires input with shape [batch_size].
        # Give output with shape [batch_size, embed_dim].
        self.label_embed = nn.Embedding(output_dim, embed_dim)

        # Acquires input with shape [batch_size, embed_dim].
        # Give output with shape [batch_size, embed_dim].
        self.context_fc = nn.Linear(embed_dim, embed_dim)

        self.out_fc = nn.Linear(embed_dim, output_dim)

    def forward(self, context_features, labels):
        '''
        :param context_features: Gathered features, of shape [batch_size, embed_dim].
        :param labels: LongTensor type labels, of shape [batch_size].
        :return out_features: features that extracted from gathered contextual features, of shape [batch_size, embed_dim].
        :return out_embeddings: embedding features that built from label embedding network, of shape [batch_size, embed_dim].
        '''

        # [batch_size, embed_dim].
        out_features = context_features

        # [batch_size, output_dim].
        out_logits = self.out_fc(out_features)

        # [batch_size, embed_dim].
        out_embeddings = self.label_embed(labels)

        return out_features, out_logits, out_embeddings, self.label_embed.weight


# The simplest linear-model, could be used in other models.
class Linear(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(Linear, self).__init__()

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(
            in_features=model_param.max_seq_len * model_param.embed_dim,
            out_features=model_param.output_dim
        )

        self.label_embedding = LabelEmbedding(model_param.max_seq_len * model_param.embed_dim,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(model_param.max_seq_len * model_param.embed_dim,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, max_seq_len, embed_dim].
        features = self.embedding(x)

        # [batch_size, max_seq_len * embed_dim].
        features = features.view(features.size()[0], -1)

        # [batch_size, max_seq_len * embed_dim].
        features = F.relu(features)

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc(features)
                out2, label_embeds, _ = self.label_embedding(features, labels)
                return out1, out2, label_embeds

            else:
                # Tuples.
                logits = self.fc(features)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(features, labels)
                return logits, out_features, out_logits, out_embeddings

        else:

            # [batch_size, output_dim].
            outputs = self.fc(features)
            return outputs


# Double-layer linear, could be used in other models.
class DoubleLinear(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(DoubleLinear, self).__init__()

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            in_features=model_param.max_seq_len * model_param.embed_dim,
            out_features=model_param.hidden_dim
        )
        self.fc2 = nn.Linear(
            in_features=model_param.hidden_dim,
            out_features=model_param.output_dim
        )

        self.label_embedding = LabelEmbedding(model_param.hidden_dim,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(model_param.hidden_dim,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, max_seq_len, embed_dim].
        features = self.embedding(x)

        # [batch_size, max_seq_len * embed_dim].
        features = features.view(features.size()[0], -1)

        # [batch_size, max_seq_len * embed_dim].
        features = F.relu(features)

        # [batch_size, hidden_dim].
        features = self.fc1(features)

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc2(features)
                out2, label_embeds, _ = self.label_embedding(features, labels)
                return out1, out2, label_embeds

            else:
                logits = self.fc2(features)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(features, labels)
                return logits, out_features, out_logits, out_embeddings

        else:

            # [batch_size, output_dim].
            outputs = self.fc2(features)
            return outputs


# Convolution network for texts.
class TextCNN(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(TextCNN, self).__init__()

        self.embed_dim = model_param.embed_dim
        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)

        self.max_seq_len = model_param.max_seq_len

        # Build with pre-trained embedding vectors, if given.
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)
            self.embedding.weight.requires_grad = False

        # Convolution layers.
        self.conv1 = nn.Conv1d(model_param.max_seq_len, model_param.max_seq_len * 2, kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(model_param.max_seq_len, model_param.max_seq_len * 2, kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(model_param.max_seq_len, model_param.max_seq_len * 2, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(model_param.max_seq_len, model_param.max_seq_len * 2, kernel_size=5, stride=2)

        # Polling layers.
        self.pool1 = nn.MaxPool1d(self._calculate_conv_feature_dim(self.embed_dim, 2, 2), 1)
        self.pool2 = nn.MaxPool1d(self._calculate_conv_feature_dim(self.embed_dim, 3, 2), 1)
        self.pool3 = nn.MaxPool1d(self._calculate_conv_feature_dim(self.embed_dim, 4, 2), 1)
        self.pool4 = nn.MaxPool1d(self._calculate_conv_feature_dim(self.embed_dim, 5, 2), 1)

        self.fc = nn.Linear(self.max_seq_len * 8, model_param.output_dim)

        self.label_embedding = LabelEmbedding(self.max_seq_len * 8, model_param.output_dim, model_param.device)
        self.use_label_embed = model_param.use_label_embed

    @staticmethod
    def _calculate_conv_feature_dim(original_dim: int, kernel_size: int, stride: int):
        return int((original_dim - kernel_size) // stride + 1)

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, max_seq_len, embed_dim].
        features = self.embedding(x)

        f1 = torch.relu(self.conv1(features))
        # [batch_size, max_seq_len * 2].
        f1p = self.pool1(f1).squeeze(2)

        f2 = torch.relu(self.conv2(features))
        # [batch_size, max_seq_len * 2].
        f2p = self.pool2(f2).squeeze(2)

        f3 = torch.relu(self.conv3(features))
        # [batch_size, max_seq_len * 2].
        f3p = self.pool3(f3).squeeze(2)

        f4 = torch.relu(self.conv4(features))
        # [batch_size, max_seq_len * 2].
        f4p = self.pool4(f4).squeeze(2)

        # [batch_size, max_seq_len * 8].
        union = torch.cat((f1p, f2p, f3p, f4p), dim=1)

        if self.use_label_embed:
            out1 = self.fc(union)
            out2, label_embed, _ = self.label_embedding(union, labels)
            return out1, out2, label_embed
        else:
            outputs = self.fc(union)
            return outputs


# Traditional simple GRU, with no attention mechanism applied.
class GRU(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(GRU, self).__init__()

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)

        # Build with pre-trained embedding vectors, if given.
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)
            self.embedding.weight.requires_grad = False

        self.rnn = nn.GRU(model_param.embed_dim,
                          model_param.hidden_dim,
                          num_layers=2,
                          bias=True,
                          batch_first=True,
                          dropout=0.5,
                          bidirectional=False)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(in_features=model_param.hidden_dim, out_features=128),
            nn.Linear(in_features=128, out_features=model_param.output_dim)
        )

        self.label_embedding = LabelEmbedding(model_param.hidden_dim,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(model_param.hidden_dim,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, max_seq_len, embed_dim].
        features = self.embedding(x)
        
        # [batch_size, max_seq_len, hidden_dim].
        outputs, _ = self.rnn(features)

        # [batch_size, hidden_dim].
        outputs = outputs[:, -1, :]

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc(outputs)
                out2, label_embeds, _ = self.label_embedding(outputs, labels)
                return out1, out2, label_embeds

            else:
                # Tuples.
                logits = self.fc(outputs)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(outputs, labels)
                return logits, out_features, out_logits, out_embeddings

        else:
            # [batch_size, output_dim].
            outputs = self.fc(outputs)
            return outputs


# Traditional BERT model, fine-tuning for the last layer.
class Bert(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(Bert, self).__init__()

        self.pre_trained_model = model_param.pre_trained_model if model_param.pre_trained_model is not None \
            else 'bert-base-uncased'

        self.encoder = BertModel.from_pretrained(self.pre_trained_model, cache_dir=model_param.pre_trained_cache)
        if model_param.freeze_pre_trained:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pre_trained_hidden_size = 768
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.pre_trained_hidden_size, out_features=model_param.hidden_dim),
            nn.Linear(in_features=model_param.hidden_dim, out_features=model_param.output_dim)
        )

        self.label_embedding = LabelEmbedding(self.pre_trained_hidden_size,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(self.pre_trained_hidden_size,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. This is suggested to be provided.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, 768], getting pooled outputs from BertModel.
        features = self.encoder(input_ids=x, attention_mask=masks)[1]

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc(features)
                out2, label_embeds, _ = self.label_embedding(features, labels)
                return out1, out2, label_embeds

            else:
                # Tuples
                logits = self.fc(features)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(features, labels)
                return logits, out_features, out_logits, out_embeddings

        else:
            # [batch_size, output_dim].
            outputs = self.fc(features)
            return outputs


# Bert with RNN Decoder.
class BertRNN(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(BertRNN, self).__init__()

        self.hidden_dim = model_param.hidden_dim
        self.device = model_param.device

        self.pre_trained_model = model_param.pre_trained_model if model_param.pre_trained_model is not None \
            else 'bert-base-uncased'

        self.encoder = BertModel.from_pretrained(self.pre_trained_model, cache_dir=model_param.pre_trained_cache)
        if model_param.freeze_pre_trained:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pre_trained_hidden_size = 768

        self.ctx = torch.zeros((2, 1, model_param.hidden_dim), dtype=torch.float).to(model_param.device)
        self.RNN = nn.GRU(input_size=self.pre_trained_hidden_size,
                          hidden_size=model_param.hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          dropout=0.5,
                          bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim * 2, out_features=model_param.hidden_dim),
            nn.Linear(in_features=model_param.hidden_dim, out_features=model_param.output_dim)
        )

        self.label_embedding = LabelEmbedding(model_param.hidden_dim * 2,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(model_param.hidden_dim * 2,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. This is suggested to be provided.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, max_seq_len, 768], getting pooled outputs from BertModel.
        features = self.encoder(input_ids=x, attention_mask=masks)[0]

        # [2, 1, hidden_dim] at first.
        out_hiddens = torch.randn((2, 1, self.hidden_dim)).to(self.device)
        out_hidden = torch.randn((2, 1, self.hidden_dim)).to(self.device)
        out_hidden.data.copy_(self.ctx)

        # [1, max_seq_len, 768].
        for feature in features:

            # [2, 1, hidden_dim].
            _, out_hidden = self.RNN(feature.unsqueeze(0), out_hidden)

            out_hiddens = torch.cat((out_hiddens, out_hidden), dim=1)

        # This time out_hiddens should be of shape [2, batch_size+1, hidden_dim].
        # Transmit it to shape [batch_size, hidden_dim * 2].
        out_hiddens = out_hiddens[:, 1:, :]
        out_hiddens = out_hiddens.transpose(0, 1).reshape(-1, self.hidden_dim * 2)

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc(out_hiddens)
                out2, label_embeds, _ = self.label_embedding(out_hiddens, labels)
                return out1, out2, label_embeds

            else:
                # Tuples.
                logits = self.fc(out_hiddens)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(out_hiddens, labels)
                return logits, out_features, out_logits, out_embeddings

        else:
            # [batch_size, output_dim].
            outputs = self.fc(out_hiddens)
            return outputs


# Distilled Bert.
class DistillBert(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(DistillBert, self).__init__()

        self.pre_trained_model = model_param.pre_trained_model if model_param.pre_trained_model is not None \
            else 'distilbert-base-uncased'

        self.encoder = DistilBertModel.from_pretrained(self.pre_trained_model, cache_dir=model_param.pre_trained_cache)
        if model_param.freeze_pre_trained:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pre_trained_hidden_size = 768
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.pre_trained_hidden_size, out_features=model_param.hidden_dim),
            nn.Linear(in_features=model_param.hidden_dim, out_features=model_param.output_dim)
        )

        self.label_embedding = LabelEmbedding(self.pre_trained_hidden_size,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(self.pre_trained_hidden_size,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Suggested to be provided.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, max_seq_len, 768], all last hidden states.
        outputs = self.encoder(input_ids=x, attention_mask=masks)[0]

        # [batch_size, 768], get last hidden states of each sequence.
        features = outputs[:, -1, :].squeeze()

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc(features)
                out2, label_embeds, _ = self.label_embedding(features, labels)
                return out1, out2, label_embeds

            else:
                # Tuples.
                logits = self.fc(features)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(features, labels)
                return logits, out_features, out_logits, out_embeddings

        else:
            # [batch_size, output_dim].
            outputs = self.fc(features)
            return outputs


# Traditional XLNet model, fine-tuning for the last layer.
# This is not implemented yet, due to some training issues.
class XlNet(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(XlNet, self).__init__()

        self.pre_trained_model = model_param.pre_trained_model if model_param.pre_trained_model is not None \
            else 'xlnet-base-cased'

        self.encoder = XLNetModel.from_pretrained(self.pre_trained_model, cache_dir=model_param.pre_trained_cache)
        if model_param.freeze_pre_trained:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.pre_trained_hidden_size = 768
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.pre_trained_hidden_size, out_features=model_param.hidden_dim),
            nn.Linear(in_features=model_param.hidden_dim, out_features=model_param.output_dim)
        )

        self.label_embedding = LabelEmbedding(self.pre_trained_hidden_size,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(self.pre_trained_hidden_size,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Suggested to be provided.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # [batch_size, max_seq_len, 768], all last hidden states.
        outputs = self.encoder(input_ids=x, attention_mask=masks)[0]

        # [batch_size, 768], get last hidden states of each sequence.
        features = outputs[:, -1, :].squeeze()

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc(features)
                out2, label_embeds, _ = self.label_embedding(features, labels)
                return out1, out2, label_embeds

            else:
                # Tuples.
                logits = self.fc(features)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(features, labels)
                return logits, out_features, out_logits, out_embeddings

        else:
            # [batch_size, output_dim].
            outputs = self.fc(features)
            return outputs


# Private module: LSTM-Encoder.
class HLstm_Encoder(nn.Module):

    def __init__(self, embed_dim: int, hidden_dim: int):
        super(HLstm_Encoder, self).__init__()

        self.rnn = nn.LSTM(embed_dim,
                           hidden_dim,
                           num_layers=2,
                           bidirectional=True,
                           batch_first=True,
                           dropout=0.5)

    def forward(self, in_feature):
        '''
        :param in_feature: torch.Tensor of shape [batch_size, max_seq_len, embed_dim].
        :return hidden_states: torch.Tensor of shape [batch_size, max_seq_len, hidden_dim].
        '''

        # [batch_size, max_seq_len, hidden_dim * 2].
        hidden_states, _ = self.rnn(in_feature)

        return hidden_states


# Private module: LSTM-Attn.
class HLstm_Attn(nn.Module):

    def __init__(self, hidden_dim: int):
        super(HLstm_Attn, self).__init__()

        self.weights = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim * 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, hidden_states):
        '''
        :param hidden_states: [batch_size, max_seq_len, hidden_dim * 2].
        :return context: [batch_size, hidden_dim * 2].
        '''

        # [batch_size, max_seq_len, hidden_dim * 2].
        query = self.dropout(self.weights(hidden_states))
        dim_k = query.size(-1)

        # [batch_size, max_seq_len, max_seq_len].
        scores = torch.matmul(query, hidden_states.transpose(1, 2)) / math.sqrt(dim_k)

        # [batch_size, max_seq_len, max_seq_len].
        p_attn = nn.functional.softmax(scores, dim=-1)

        # [batch_size, hidden_dim * 2].
        context = torch.matmul(p_attn, hidden_states).sum(dim=1)

        return context


# Private module: LSTM-Decoder.
class HLstm_Decoder(nn.Module):

    def __init__(self, hidden_dim: int, concat_len: int):
        super(HLstm_Decoder, self).__init__()

        self.rnn = nn.LSTM(hidden_dim * (concat_len * 3 + 2),
                           hidden_dim,
                           num_layers=1,
                           bidirectional=False,
                           batch_first=True,
                           dropout=0.5)

    def forward(self, gather_feature):
        '''
        :param gather_feature: [1, 1, hidden_dim * 5 or 8].
        :return outputs: [1, hidden_dim].
        '''

        # [1, 1, hidden_dim].
        outputs, _ = self.rnn(gather_feature)

        return F.relu(outputs.squeeze(0))


# Hierarchical Bi-LSTM.
# Implementation of paper 'Rare-Class Dialogue Act Tagging for Alzheimer’s Disease Diagnosis'
class HLstm(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(HLstm, self).__init__()

        self.embed_dim = model_param.embed_dim
        self.hidden_dim = model_param.hidden_dim
        self.concat_len = 1
        self.device = model_param.device

        self.embedding = nn.Embedding(model_param.vocab_size, self.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.encoder = HLstm_Encoder(self.embed_dim, self.hidden_dim).to(self.device)
        self.attention = HLstm_Attn(self.hidden_dim).to(self.device)
        self.decoder = HLstm_Decoder(self.hidden_dim, self.concat_len).to(self.device)

        self.fc = nn.Sequential(
            nn.Linear(
                in_features=model_param.hidden_dim,
                out_features=model_param.hidden_dim
            ),
            nn.Linear(
                in_features=model_param.hidden_dim,
                out_features=model_param.output_dim
            )
        )
        
        self.l1 = torch.randn((1, model_param.hidden_dim)).to(model_param.device)
        self.l2 = torch.randn((1, model_param.hidden_dim)).to(model_param.device)
        self.c1 = torch.randn((1, model_param.hidden_dim * 2)).to(model_param.device)
        self.c2 = torch.randn((1, model_param.hidden_dim * 2)).to(model_param.device)

        self.label_embedding = LabelEmbedding(model_param.hidden_dim,
                                              model_param.output_dim,
                                              model_param.device)
        self.context_label_embed = ContextLabelEmbed(model_param.hidden_dim,
                                                     model_param.output_dim,
                                                     model_param.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # Read batch_size.
        batch_size = x.size(0)

        # [batch_size, max_seq_len, embed_dim].
        # Use word embeddings to extract the lexical feature representations from the transcripts.
        features = self.embedding(x)

        # [1, hidden_dim] at first.
        label_features = torch.randn((1, self.hidden_dim)).to(self.device)

        # [1, hidden_dim * 2] at first. Only used for some testing (abandoned).
        context_features = torch.randn((1, self.hidden_dim * 2)).to(self.device)

        # Extract one piece of features each time.
        # In this case, all features that fed into relative sub-modules keep their size of batch being 1.
        for feature in features:

            # [1, max_seq_len, embed_dim].
            # in_feature represents an utterance.
            in_feature = feature.unsqueeze(0)

            # [1, max_seq_len, hidden_dim * 2].
            # Word representation layer feeds into a BiLSTM, producing a sequence of hidden vectors.
            hidden_states = self.encoder(in_feature)

            # [1, hidden_dim * 2].
            # Use attention mechanism to weight these hidden vectors.
            # Then aggregate them into a single utterance representation.
            context = self.attention(hidden_states)

            if self.concat_len == 1:

                # [1, hidden_dim * 6].
                # Concatenate vectors from the past and now.
                gather_feature = torch.cat((self.l2, self.c2, context), 1).unsqueeze(0)

                # [1, hidden_dim].
                # These concatenated vectors are then encoded by a second unidirectional LSTM.
                decoder_feature = self.decoder(gather_feature)

                label_features = torch.cat((label_features, decoder_feature), 0)
                context_features = torch.cat((context_features, context), 0)

                # Converting context information.
                self.c2.data.copy_(context)

                # Converting label information.
                self.l2.data.copy_(decoder_feature)

            else:

                # [1, hidden_dim * 8].
                # Concatenate vectors from the past and now.
                gather_feature = torch.cat((self.l1, self.c1, self.l2, self.c2, context), 1).unsqueeze(0)

                # [1, hidden_dim].
                # These concatenated vectors are then encoded by a second unidirectional LSTM.
                decoder_feature = self.decoder(gather_feature)

                label_features = torch.cat((label_features, decoder_feature), 0)
                context_features = torch.cat((context_features, context), 0)

                # Converting context information.
                self.c1.data.copy_(self.c2)
                self.c2.data.copy_(context)

                # Converting label information.
                self.l1.data.copy_(self.l2)
                self.l2.data.copy_(decoder_feature)

        # [batch_size, hidden_dim].
        outputs = label_features[1:]

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuples.
                out1 = self.fc(outputs)
                out2, label_embeds, _ = self.label_embedding(outputs, labels)
                return (out1, out2, label_embeds)

            else:
                # Tuples.
                logits = self.fc(outputs)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(outputs, labels)
                return logits, out_features, out_logits, out_embeddings

        else:
            # [batch_size, output_dim].
            outputs = self.fc(outputs)
            return outputs


# Private module: Positional encoder.
# This module is used for convert embedded features to their positional-encoded form.
class PositionEncoder(nn.Module):
    r"""
        Usage example::

        >>> batch_size = 32
        >>> max_seq_len = 25
        >>> d_model = 300
        >>> x = torch.randn((batch_size, max_seq_len, d_model))
        >>> position_encoder = PositionEncoder(max_seq_len, d_model)
        >>> px = position_encoder(x)     # [batch_size, max_seq_len, d_model]
    """

    def __init__(self, max_seq_len, d_model, device=torch.device('cpu')):
        '''
        :param max_seq_len: Fixed length of input sequences, of type int.
        :param d_model: Fixed dimension of input sequences, of type int.
        :param device: Designated torch.device.
        '''
        super(PositionEncoder, self).__init__()

        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.device = device

        self.dropout = nn.Dropout(0.5)
        self.pe = torch.zeros((max_seq_len, d_model)).to(device)
        self._init_pe()

        # pe.size after _init_pe() called: [1, max_seq_len, d_model].

    # Initialize positional encoding data.
    def _init_pe(self):

        position = torch.arange(0., self.max_seq_len).to(self.device)
        div_term = torch.exp(
            torch.arange(0., self.d_model, 2) * -(math.log(10000.0) / (self.d_model))
        ).to(self.device)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe.unsqueeze_(0)

    def forward(self, x):
        '''
        :param x: Sequences to positional encode on, size is [batch_size, max_seq_len, d_model].
        :return outputs: Sequences that has been positional encoded, size is [batch_size, max_seq_len, d_model].
        '''

        # [batch_size, max_seq_len, d_model].
        outputs = self.dropout(x + self.pe[:, :x.size(1)])

        return outputs


# Private module: Scaled dot product attention.
# This module produces attention scores and attention context, without any extra weights that used for computing.
class ScaledDotProductAttn(nn.Module):
    r"""
        Usage Example::

        >>> d_model = 300
        >>> x = torch.randn((64, 25, d_model))
        >>> wq = nn.Linear(d_model, d_model)
        >>> wk = nn.Linear(d_model, d_model)
        >>> wv = nn.Linear(d_model, d_model)
        >>> Q = wq(x)
        >>> K = wk(x)
        >>> V = wv(x)
        >>> attention = ScaledDotProductAttn(d_model)
        >>> context, attn = attention(Q, K, V)
    """

    def __init__(self, d_model, device=torch.device('cpu')):
        super(ScaledDotProductAttn, self).__init__()

        self.d_model = d_model
        self.device = device

    def forward(self, Q, K, V, mask=None):
        '''
        :param Q: [..., seq_len, d_model].
        :param K: [..., seq_len, d_model].
        :param V: [..., seq_len, d_model].
        :param mask: [..., seq_len, seq_len].
        :return context: Context vector, of shape [..., seq_len, d_model].
        :return attn: Attention weights, of shape [..., seq_len, seq_len].
        '''

        # [..., seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_model)

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.bool).to(self.device)
            scores.masked_fill_(mask, -1e9)

        # [..., seq_len, seq_len].
        attn = nn.Softmax(dim=-1)(scores)

        # [..., seq_len, d_model].
        context = torch.matmul(attn, V)

        return context, attn


# Private module: Scaled attention mechanism with multi-heads.
# This module calculates attention scores and attention context in multi-head structure.
# It will make use of ScaledDotProductAttn module.
class WeightedMultiHeadAttn(nn.Module):
    r"""
        Usage example::

        >>> num_heads = 2
        >>> d_model = 300
        >>> x = torch.randn((64, 25, d_model))
        >>> wq = nn.Linear(d_model, d_model)
        >>> wk = nn.Linear(d_model, d_model)
        >>> wv = nn.Linear(d_model, d_model)
        >>> Q = wq(x)
        >>> K = wk(x)
        >>> V = wv(x)
        >>> attention = WeightedMultiHeadAttn(num_heads, d_model)
        >>> context, attn = attention(Q, K, V)
    """

    def __init__(self, num_heads, d_model, device=torch.device('cpu')):
        super(WeightedMultiHeadAttn, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.device = device

        self.WQ = nn.Linear(d_model, num_heads * d_model).to(device)
        self.WK = nn.Linear(d_model, num_heads * d_model).to(device)
        self.WV = nn.Linear(d_model, num_heads * d_model).to(device)

        self.dot_attn = ScaledDotProductAttn(d_model, device).to(device)
        self.fc = nn.Linear(num_heads * d_model, d_model).to(device)
        self.layer_norm = nn.LayerNorm(d_model).to(device)

    def forward(self, Q, K, V, mask=None):
        '''
        :param Q: [batch_size, seq_len, d_model].
        :param K: [batch_size, seq_len, d_model].
        :param V: [batch_size, seq_len, d_model].
        :param mask: [batch_size, seq_len, seq_len].
        :return context: [batch_size, seq_len, d_model].
        :return attn: Attention weights that have been soft-maxed. [batch_size, num_heads, seq_len, seq_len].
        '''

        batch_size = Q.size(0)
        seq_len = Q.size(1)

        # [batch_size, num_heads, seq_len, d_model].
        weighted_Q = self.WQ(Q).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1, 2)
        weighted_K = self.WK(K).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1, 2)
        weighted_V = self.WV(V).view(batch_size, seq_len, self.num_heads, self.d_model).transpose(1, 2)

        # [batch_size, num_heads, seq_len, seq_len].
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # context: [batch_size, num_heads, seq_len, d_model].
        # attn: [batch_size, num_heads, seq_len, seq_len].
        context, attn = self.dot_attn(weighted_Q, weighted_K, weighted_V, mask)

        # [batch_size, seq_len, num_heads * d_model].
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_model)

        # [batch_size, seq_len, d_model].
        context = self.layer_norm(self.fc(context) + Q)

        return context, attn


# Sequence-to-sequence with no attention.
class Seq2Seq(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(Seq2Seq, self).__init__()

        self.hidden_dim = model_param.hidden_dim
        self.device = model_param.device

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.window_size = 5
        self.forehead_states = torch.randn((self.window_size, self.hidden_dim * 2)).to(self.device)

        self.token_encoder = nn.GRU(input_size=model_param.embed_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)
        self.utt_encoder = nn.GRU(input_size=self.hidden_dim * 2,
                                  hidden_size=self.hidden_dim,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)

        self.vanilla_decoder1 = nn.GRU(input_size=model_param.embed_dim,
                                       hidden_size=self.hidden_dim,
                                       num_layers=1,
                                       batch_first=True,
                                       bidirectional=False)
        self.vanilla_decoder2 = nn.GRU(input_size=self.hidden_dim,
                                       hidden_size=self.hidden_dim,
                                       num_layers=1,
                                       batch_first=True,
                                       bidirectional=False)

        self.decoder = nn.GRU(input_size=self.hidden_dim,
                              hidden_size=self.hidden_dim * 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=False)

        self.fc = nn.Linear(in_features=self.hidden_dim * 2, out_features=model_param.output_dim)

        self.label_embedding = LabelEmbedding(model_param.hidden_dim * 2,
                                              model_param.output_dim,
                                              model_param.device)
        self.use_label_embed = model_param.use_label_embed

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        batch_size = x.size(0)

        # [batch_size, max_seq_len, embed_dim].
        token_features = self.embedding(x)

        # [2, batch_size, hidden_dim].
        # The hidden states of bi-GRU encoder represents utterance features.
        _, utt_features = self.token_encoder(token_features)

        # [batch_size, hidden_dim * 2].
        utt_features = utt_features.transpose(0, 1).reshape((-1, self.hidden_dim * 2))

        # [batch_size, max_seq_len, hidden_dim].
        vanilla_features1, _ = self.vanilla_decoder1(token_features)

        # [batch_size, max_seq_len, hidden_dim].
        vanilla_features2, _ = self.vanilla_decoder2(vanilla_features1)

        # [2, batch_size, hidden_dim].
        context_features = torch.zeros((2, batch_size, self.hidden_dim), dtype=torch.float).to(self.device)

        # Input utterance features one-by-one.
        for idx in range(batch_size):

            for window_idx in range(self.window_size):

                if idx + window_idx >= batch_size:
                    break

                # Slicing window.
                self.forehead_states[window_idx].data.copy_(utt_features[idx + window_idx])

            # [2, 1, hidden_dim].
            _, context_feature = self.utt_encoder(self.forehead_states.unsqueeze(0))

            context_features[:, idx, :].data.copy_(context_feature.squeeze(1))

        # [1, batch, hidden_dim * 2].
        cat_context_features = torch.cat((context_features[0], context_features[1]), dim=1).unsqueeze(0)

        # [batch_size, max_seq_len, hidden_dim * 2].
        features, _ = self.decoder(vanilla_features2, cat_context_features)

        # Use the last hidden states as output features.
        # [batch_size, hidden_dim * 2].
        features = features[:, -1, :]

        if self.use_label_embed:
            out1 = self.fc(features)
            out2, label_embed, _ = self.label_embedding(features, labels)
            return out1, out2, label_embed

        else:
            outputs = self.fc(features)
            return outputs


# Sequence-to-sequence with attention.
class Seq2SeqAttn(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(Seq2SeqAttn, self).__init__()

        self.hidden_dim = model_param.hidden_dim
        self.device = model_param.device

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.encoder1 = nn.GRU(input_size=model_param.embed_dim,
                               hidden_size=model_param.hidden_dim,
                               batch_first=True,
                               num_layers=1,
                               bidirectional=True)
        self.encoder2 = nn.GRU(input_size=model_param.hidden_dim * 2,
                               hidden_size=model_param.hidden_dim,
                               batch_first=True,
                               num_layers=1,
                               bidirectional=True)
        self.vanilla_encoder1 = nn.GRU(input_size=model_param.embed_dim,
                                      hidden_size=model_param.hidden_dim,
                                      batch_first=True,
                                      num_layers=1,
                                      bidirectional=False)
        self.vanilla_encoder2 = nn.GRU(input_size=model_param.hidden_dim,
                                       hidden_size=model_param.hidden_dim,
                                       batch_first=True,
                                       num_layers=1,
                                       bidirectional=False)
        self.decoder = nn.GRU(input_size=model_param.hidden_dim * 2,
                              hidden_size=model_param.hidden_dim * 2,
                              batch_first=True,
                              num_layers=1,
                              bidirectional=False)

        self.window_size = 5
        self.forehead_states = torch.randn((self.window_size, self.hidden_dim * 2)).to(self.device)

        self.attention_type = 'vanilla'
        self.attention = ScaledDotProductAttn(self.hidden_dim * 2, self.device)
        self.attn_relu = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(model_param.hidden_dim * 2, model_param.output_dim)

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        batch_size = x.size(0)

        # [batch_size, max_seq_len, embed_dim].
        token_features1 = self.embedding(x)

        # [batch_size, max_seq_len, hidden_dim * 2].
        token_features2, _ = self.encoder1(token_features1)

        # [batch_size, max_seq_len, hidden_dim].
        vanilla_features1, _ = self.vanilla_encoder1(token_features1)

        # [batch_size, max_seq_len, hidden_dim].
        vanilla_features2, _ = self.vanilla_encoder2(vanilla_features1)

        # [2, batch_size, hidden_dim].
        _, utt_features = self.encoder2(token_features2)

        # [batch_size, hidden_dim * 2].
        utt_features = torch.cat((utt_features[0], utt_features[1]), dim=1)

        context_features = torch.randn((1, self.hidden_dim * 2)).to(self.device)

        # Slice window.
        for batch_idx in range(batch_size):

            for window_idx in range(self.window_size - 1):
                self.forehead_states[window_idx].data.copy_(self.forehead_states[window_idx + 1])
            self.forehead_states[self.window_size - 1].data.copy_(utt_features[batch_idx])

            # [1, window_size, hidden_dim * 2].
            window_features, _ = self.decoder(self.forehead_states.unsqueeze(0))

            if self.attention_type == 'vanilla':

                # [1, window_size, window_size].
                _, attn_scores = self.attention(self.forehead_states.unsqueeze(0),
                                                self.forehead_states.unsqueeze(0),
                                                window_features)
                attn_scores = self.attn_relu(attn_scores)

                # [1, hidden_dim * 2].
                context = torch.sum(torch.matmul(attn_scores, window_features), dim=1)
                context_features = torch.cat((context_features, context), dim=0)

            elif self.attention_type == 'hard':

                context_features = torch.cat((context_features, window_features[:, -1, :]), dim=0)

        context_features = context_features[1:]

        # [batch_size, output_dim].
        outputs = self.fc(context_features)
        return outputs


# Window-treated Seq2Seq.
# In this model, batch_size will be treated as window_size.
class WinSeq2Seq(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(WinSeq2Seq, self).__init__()

        self.hidden_dim = model_param.hidden_dim
        self.output_dim = model_param.output_dim
        self.device = model_param.device

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.token_encoder = nn.GRU(input_size=model_param.embed_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)

        self.utt_encoder = nn.GRU(input_size=self.hidden_dim * 2,
                                  hidden_size=self.hidden_dim,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)

        # Acquires input features with shape [batch_size, max_seq_len, hidden_dim * 2].
        # Acquires input hidden states with shape [1, batch_size, hidden_dim * 2].
        self.decoder = nn.GRU(input_size=self.hidden_dim * 2,
                              hidden_size=self.hidden_dim * 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=False)

        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)

        self.label_embedding = LabelEmbedding(self.hidden_dim * 2, self.output_dim, self.device)
        self.use_label_embed = model_param.use_label_embed

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [window_size, max_seq_len].
            :param labels: torch.Tensor, of shape [window_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [window_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [window_size, output_dim].
        '''

        window_size = x.size(0)

        # [window_size, max_seq_len, embed_dim].
        token_features = self.embedding(x)

        # [2, window_size, hidden_dim].
        _, utt_features = self.token_encoder(token_features)

        # [1, window_size, hidden_dim * 2].
        utt_features = utt_features.transpose(0, 1).reshape((1, window_size, -1))

        # [2, 1, hidden_dim].
        _, context = self.utt_encoder(utt_features)

        # [1, 1, hidden_dim * 2].
        context = context.transpose(0, 1).reshape((1, 1, -1))

        decoder_input = torch.zeros((1, 1, self.hidden_dim * 2), dtype=torch.float).to(self.device)
        decoder_outputs = torch.randn((1, self.hidden_dim * 2)).to(self.device)

        for i in range(window_size):

            decoder_input, context = self.decoder(decoder_input, context)
            decoder_outputs = torch.cat((decoder_outputs, decoder_input.squeeze(0)), dim=0)

        # [window_size, hidden_dim * 2].
        decoder_outputs = decoder_outputs[1:]

        if self.use_label_embed:
            out1 = self.fc(decoder_outputs)
            out2, label_embed, _ = self.label_embedding(decoder_outputs, labels)
            return out1, out2, label_embed
        else:
            outputs = self.fc(decoder_outputs)
            return outputs


# Window-treated Seq2Seq.
# In this model, batch_size will be treated as window_size.
# This model follows paper 'Guiding attention in Sequence to sequence model for Dialogue Act prediction'.
# Context vector will be computed in different form.
class WinSeq2SeqAttn(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(WinSeq2SeqAttn, self).__init__()

        self.hidden_dim = model_param.hidden_dim
        self.output_dim = model_param.output_dim
        self.device = model_param.device

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.token_encoder = nn.GRU(input_size=model_param.embed_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True)
        self.utt_encoder = nn.GRU(input_size=self.hidden_dim * 2,
                                  hidden_size=self.hidden_dim,
                                  num_layers=1,
                                  batch_first=True,
                                  bidirectional=True)

        # Acquires input features with shape [batch_size, max_seq_len, hidden_dim * 2].
        # Acquires input hidden states with shape [1, batch_size, hidden_dim * 2].
        self.decoder = nn.GRU(input_size=self.hidden_dim * 2,
                              hidden_size=self.hidden_dim * 2,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=False)

        self.fc = nn.Linear(self.hidden_dim * 2, self.output_dim)

        self.label_embedding = LabelEmbedding(self.hidden_dim * 2, self.output_dim, self.device)
        self.context_label_embed = ContextLabelEmbed(self.hidden_dim * 2, self.output_dim, self.device)
        self.use_label_embed = model_param.use_label_embed
        self.label_embed_ver = model_param.label_embed_ver

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [window_size, max_seq_len].
            :param labels: torch.Tensor, of shape [window_size].
            :param masks: torch.Tensor, of shape [window_size, max_seq_len].
            :return outputs: torch.Tensor, of shape [window_size, output_dim].
        '''

        window_size = x.size(0)

        # [window_size, max_seq_len, embed_dim].
        token_features = self.embedding(x)

        # [2, window_size, hidden_dim].
        _, utt_features = self.token_encoder(token_features)

        # [1, window_size, hidden_dim * 2].
        utt_features = utt_features.transpose(0, 1).reshape((1, window_size, -1))

        # [1, window_size, hidden_dim * 2].
        utt_outputs, _ = self.utt_encoder(utt_features)

        decoder_input = torch.zeros((1, 1, self.hidden_dim * 2), dtype=torch.float).to(self.device)
        decoder_outputs = torch.randn((1, self.hidden_dim * 2)).to(self.device)

        for i in range(window_size):

            decoder_input, _ = self.decoder(decoder_input, utt_outputs[:, i, :].unsqueeze(1))
            decoder_outputs = torch.cat((decoder_outputs, decoder_input.squeeze(0)), dim=0)

        # [window_size, hidden_dim * 2].
        decoder_outputs = decoder_outputs[1:]

        if self.use_label_embed:

            if self.label_embed_ver == 1:
                # Tuple.
                out1 = self.fc(decoder_outputs)
                out2, label_embed, _ = self.label_embedding(decoder_outputs, labels)
                return out1, out2, label_embed

            else:
                # Tuple.
                logits = self.fc(decoder_outputs)
                out_features, out_logits, out_embeddings, _ = self.context_label_embed(decoder_outputs, labels)
                return logits, out_features, out_logits, out_embeddings

        else:
            outputs = self.fc(decoder_outputs)
            return outputs


# Private module: Scaled attention mechanism with multi-heads.
class ScaledMultiHeadAttn(nn.Module):

    def __init__(self):
        super(ScaledMultiHeadAttn, self).__init__()

    def forward(self, q, k, v, mask=None):
        '''
        :param q: [batch_size, num_heads, seq_len, depth].
        :param k: [batch_size, num_heads, seq_len, depth].
        :param v: [batch_size, num_heads, seq_len, depth].
        :param mask: [batch_size, num_heads, seq_len, seq_len].
        :return attn_weights: [batch_size, num_heads, seq_len, seq_len].
        :return outputs: [batch_size, num_heads, seq_len, depth].
        '''

        batch_size = q.size(0)
        num_heads = q.size(1)
        seq_len = q.size(2)
        depth = q.size(3)

        # [batch_size * num_heads, seq_len, depth].
        q_zip = q.view(batch_size * num_heads, seq_len, depth)
        k_zip = k.view(batch_size * num_heads, seq_len, depth)
        v_zip = v.view(batch_size * num_heads, seq_len, depth)

        # [batch_size * num_heads, seq_len, seq_len].
        matmul_qk = torch.matmul(q_zip, k_zip.transpose(1, 2)) / math.sqrt(depth)

        if mask is not None:

            # [batch_size * num_heads, seq_len, seq_len].
            mask_zip = mask.view(batch_size * num_heads, seq_len, seq_len)

            # [batch_size * num_heads, seq_len, seq_len].
            matmul_qk += (mask_zip * -1e9)

        # [batch_size * num_heads, seq_len, seq_len].
        # Do softmax on the last dimension.
        attn_weights = nn.functional.softmax(matmul_qk, dim=-1)

        # [batch_size * num_heads, seq_len, depth].
        outputs = torch.matmul(attn_weights, v_zip)

        return attn_weights.view(batch_size, num_heads, seq_len, seq_len), \
               outputs.view(batch_size, num_heads, seq_len, depth)


# Private module: Hand-written Multi-head attention.
class MultiHeadAttn(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(MultiHeadAttn, self).__init__()

        assert model_param.hidden_dim * 2 % model_param.num_heads == 0, \
            'Given dimension of hidden states * 2 must be divisible by given number of attn heads!'

        self.hidden_dim = model_param.hidden_dim
        self.num_heads = model_param.num_heads
        self.depth = self.hidden_dim * 2 // self.num_heads

        self.wq = nn.Linear(
            in_features=model_param.hidden_dim * 2,
            out_features=model_param.hidden_dim * 2
        )
        self.wk = nn.Linear(
            in_features=model_param.hidden_dim * 2,
            out_features=model_param.hidden_dim * 2
        )
        self.wv = nn.Linear(
            in_features=model_param.hidden_dim * 2,
            out_features=model_param.hidden_dim * 2
        )

        self.scaled_attn = ScaledMultiHeadAttn()

        self.fc = nn.Linear(
            in_features=model_param.hidden_dim * 2,
            out_features=model_param.hidden_dim * 2
        )

    def _split_heads(self, features):
        '''
        :param features: torch.Tensor, of shape [batch_size, seq_len (could be different), hidden_dim * 2].
        :return split_features: torch.Tensor, of shape [batch_size, num_heads, seq_len, depth].
        '''

        batch_size = features.size(0)
        seq_len = features.size(1)

        # [batch_size, num_heads, seq_len, depth].
        split_features = features.view((batch_size, seq_len, self.num_heads, self.depth)).permute((0, 2, 1, 3))

        return split_features

    def forward(self, q, k, v, mask=None):
        '''
        :param q: torch.Tensor, of shape [batch_size, seq_len, hidden_dim * 2].
        :param k: torch.Tensor, of shape [batch_size, seq_len, hidden_dim * 2].
        :param v: torch.Tensor, of shape [batch_size, seq_len, hidden_dim * 2].
        :param mask: torch.Tensor, of shape [batch_size, num_heads, seq_len, seq_len].
        :return outputs: torch.Tensor, of shape [batch_size, seq_len, hidden_dim * 2].
        :return attn_products: torch.Tensor, of shape [batch_size, num_heads, seq_len, depth].
        '''

        batch_size = q.size(0)
        seq_len = q.size(1)

        # [batch_size, seq_len, hidden_dim * 2].
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # [batch_size, num_heads, seq_len, depth].
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # [batch_size, num_heads, seq_len, seq_len].
        # [batch_size, num_heads, seq_len, depth].
        attn_weights, attn_products = self.scaled_attn(q, k, v, mask)

        # [batch_size, seq_len, num_heads, depth].
        attn_products = attn_products.permute((0, 2, 1, 3))

        # [batch_size, seq_len, hidden_dim * 2].
        concat_attn = attn_products.reshape((batch_size, seq_len, -1))

        # [batch_size, seq_len, hidden_dim * 2].
        outputs = self.fc(concat_attn)

        return outputs, attn_products


# Private module: Hierarchical Label Structured Layer. One utterance for each input.
class HLSL(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(HLSL, self).__init__()

        self.multi_attn = MultiHeadAttn(model_param)
        self.ctx_encoder = nn.LSTM(model_param.hidden_dim*2,
                                   model_param.hidden_dim,
                                   num_layers=2,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=0.5)

        # Feed forward layer for concatenation.
        self.concatenate = nn.Linear(model_param.hidden_dim * 6, model_param.hidden_dim * 2)

        # Pooling method for concatenation.
        # self.concatenate = nn.MaxPool1d(3, stride=3)

    def forward(self, hidden, coarse_feature, fine_feature):
        '''
        :param hidden: torch.tensor, of shape [1, batch_size, hidden_dim * 2].
        :param coarse_feature: torch.tensor, of shape [1, batch_size, hidden_dim * 2].
        :param fine_feature: torch.tensor, of shape [1, batch_size, hidden_dim * 2].
        :return outputs: torch.tensor, of shape [1, batch_size, hidden_dim * 2], the final representation for each HLSL.
        '''

        # [1, batch_size, hidden_dim * 2].
        # First context encoder is a conversational-level encoding layer.
        # This output is Hd in the paper.
        ctx, _ = self.ctx_encoder(hidden)

        # [1, batch_size, hidden_dim * 2].
        # Course-grained label attention uses multi-head attention to incorporate coarse-grained DAs.
        # This output is Hc in the paper.
        attn1, _ = self.multi_attn(ctx, coarse_feature, coarse_feature, None)

        # [1, batch_size, hidden_dim * 2].
        # Fine-grained label attention uses coarse-grained vectors to perform attention over fine-grained DA embeddings.
        # This output is Hf in the paper.
        attn2, _ = self.multi_attn(attn1, fine_feature, fine_feature, None)

        # [1, batch_size, batch_size].
        alpha = F.softmax(torch.matmul(attn2, hidden.transpose(1, 2)), dim=-1)

        # [1, batch_size, hidden_dim * 2].
        # Attention re-weighting uses the fine-grained DA vectors to re-weight attention.
        # This output is Hu in the paper.
        re_weight = torch.matmul(alpha, hidden)

        # [1, batch_size, hidden_dim * 6].
        outputs = torch.cat((re_weight, ctx, attn2), dim=2)

        return self.concatenate(outputs.squeeze(0)).unsqueeze(0)


# Model from paper Balance the Labels Hierarchical Label Structured Network for Dialogue Act Recognition.
# This model achieves bad performance under my own implementation.
# I will fix it later.
class HLSN(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(HLSN, self).__init__()

        self.embedding = nn.Embedding(model_param.vocab_size, model_param.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.coarse_embedding = nn.Embedding(model_param.output_dim, model_param.hidden_dim * 2)
        self.fine_embedding = nn.Embedding(model_param.output_dim, model_param.hidden_dim * 2)

        self.encoder = HLstm_Encoder(model_param.embed_dim, model_param.hidden_dim).to(model_param.device)
        self.last_hidden_state = torch.randn((1, model_param.max_seq_len, model_param.hidden_dim * 2)). \
            to(model_param.device)

        self.hlsl_stacks = [HLSL(model_param).to(model_param.device) for _ in range(5)]

        self.coarse_fc = nn.Linear(
            in_features=model_param.hidden_dim * 2,
            out_features=model_param.output_dim
        )
        self.fine_fc = nn.Linear(
            in_features=model_param.hidden_dim * 2,
            out_features=model_param.output_dim
        )

        self.lam = nn.parameter.Parameter(data=torch.randn((1, 1)), requires_grad=True).to(model_param.device)
        nn.init.uniform_(self.lam, -0.1, 0.1)

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        max_seq_len = x.size(1)

        # [batch_size, max_seq_len, embed_dim].
        # Use word embeddings to extract the lexical feature representations from the transcripts.
        features = self.embedding(x)

        # [batch_size, max_seq_len, hidden_dim * 2].
        hidden_states = self.encoder(features)

        # [1, batch_size, hidden_dim * 2].
        hn = (torch.sum(hidden_states, 1) / max_seq_len).unsqueeze(0)

        # [1, batch_size, hidden_dim * 2].
        coarse_features = self.coarse_embedding(labels.unsqueeze(0))

        # [1, batch_size, hidden_dim * 2].
        fine_features = self.fine_embedding(labels.unsqueeze(0))

        # [1, batch_size, hidden_dim * 2].
        for single_hlsl in self.hlsl_stacks:
            hn = single_hlsl(hn, coarse_features, fine_features)

        # [batch_size, hidden_dim * 2].
        hn.squeeze_(0)

        # [batch_size, hidden_dim * 2].
        coarse_outputs = self.coarse_fc(hn)
        fine_outputs = self.fine_fc(hn)

        return F.softmax((fine_outputs + coarse_outputs * self.lam), dim=-1)


# Only used for testing.
# This may be the implementation of other models, with some differents.
class TestNet(nn.Module):

    def __init__(self, model_param: ModelParam):
        super(TestNet, self).__init__()

        self.embed_dim = model_param.embed_dim
        self.hidden_dim = model_param.hidden_dim
        self.concat_len = 1
        self.device = model_param.device

        self.embedding = nn.Embedding(model_param.vocab_size, self.embed_dim)
        if model_param.vocab_embedding is not None:
            self.embedding.weight.data.copy_(model_param.vocab_embedding)

        self.encoder = HLstm_Encoder(self.embed_dim, self.hidden_dim).to(self.device)
        self.attention = HLstm_Attn(self.hidden_dim).to(self.device)
        self.decoder = HLstm_Decoder(self.hidden_dim, self.concat_len).to(self.device)

        self.fc = nn.Linear(
            in_features=model_param.hidden_dim,
            out_features=model_param.output_dim
        )

        self.test_fc = nn.Sequential(
            nn.Linear(in_features=model_param.hidden_dim * (self.concat_len * 3 + 2), out_features=model_param.hidden_dim),
            nn.Linear(in_features=model_param.hidden_dim, out_features=model_param.output_dim)
        )

        self.l1 = torch.randn((1, model_param.hidden_dim)).to(model_param.device)
        self.l2 = torch.randn((1, model_param.hidden_dim)).to(model_param.device)
        self.c1 = torch.randn((1, model_param.hidden_dim * 2)).to(model_param.device)
        self.c2 = torch.randn((1, model_param.hidden_dim * 2)).to(model_param.device)

    def forward(self, x, labels=None, masks=None):
        '''
            :param x: torch.LongTensor, of shape [batch_size, max_seq_len].
            :param labels: torch.Tensor, of shape [batch_size]. Not used in this model.
            :param masks: torch.Tensor, of shape [batch_size, max_seq_len]. Not used in this model.
            :return outputs: torch.Tensor, of shape [batch_size, output_dim].
        '''

        # Read batch_size.
        batch_size = x.size(0)

        # [batch_size, max_seq_len, embed_dim].
        # Use word embeddings to extract the lexical feature representations from the transcripts.
        features = self.embedding(x)

        # [1, hidden_dim] at first.
        label_features = torch.randn((1, self.hidden_dim)).to(self.device)

        # [1, hidden_dim * 2] at first. Only used for some testing (abandoned).
        context_features = torch.randn((1, self.hidden_dim * 2)).to(self.device)

        # [1, hidden_dim * 5] at first. Used for testing.
        test_features = torch.randn((1, self.hidden_dim * (self.concat_len * 3 + 2))).to(self.device)

        # Extract one piece of features each time.
        # In this case, all features that fed into relative sub-modules keep their size of batch being 1.
        for feature in features:

            # [1, max_seq_len, embed_dim].
            # in_feature represents an utterance.
            in_feature = feature.unsqueeze(0)

            # [1, max_seq_len, hidden_dim * 2].
            # Word representation layer feeds into a BiLSTM, producing a sequence of hidden vectors.
            hidden_states = self.encoder(in_feature)

            # [1, hidden_dim * 2].
            # Use attention mechanism to weight these hidden vectors.
            # Then aggregate them into a single utterance representation.
            context = self.attention(hidden_states)

            if self.concat_len == 1:

                # [1, hidden_dim * 5].
                # Concatenate vectors from the past and now.
                gather_feature = torch.cat((self.l2, self.c2, context), 1).unsqueeze(0)

                # Gathering context features.
                test_features = torch.cat((test_features, gather_feature.squeeze(0)), 0)

                # [1, hidden_dim].
                # These concatenated vectors are then encoded by a second unidirectional LSTM.
                decoder_feature = self.decoder(gather_feature)

                label_features = torch.cat((label_features, decoder_feature), 0)
                context_features = torch.cat((context_features, context), 0)

                # Converting context information.
                self.c2.data.copy_(context)

                # Converting label information.
                self.l2.data.copy_(decoder_feature)

            else:

                # [1, hidden_dim * 8].
                # Concatenate vectors from the past and now.
                gather_feature = torch.cat((self.l1, self.c1, self.l2, self.c2, context), 1).unsqueeze(0)

                # [1, hidden_dim].
                # These concatenated vectors are then encoded by a second unidirectional LSTM.
                decoder_feature = self.decoder(gather_feature)

                label_features = torch.cat((label_features, decoder_feature), 0)
                context_features = torch.cat((context_features, context), 0)

                # Converting context information.
                self.c1.data.copy_(self.c2)
                self.c2.data.copy_(context)

                # Converting label information.
                self.l1.data.copy_(self.l2)
                self.l2.data.copy_(decoder_feature)

        # [batch_size, output_dim].
        # outputs = self.fc(label_features[1:])
        outputs = self.test_fc(test_features[1:])

        return outputs