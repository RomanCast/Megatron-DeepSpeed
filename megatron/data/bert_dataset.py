# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BERT Style dataset."""
import time

import numpy as np
import torch

from megatron import (
    get_args,
    get_tokenizer,
    mpu,
    print_rank_0
)
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import (
    get_samples_mapping,
    get_a_and_b_segments,
    get_split_by_range_,
    truncate_segments,
    create_tokens_and_tokentypes,
    create_masked_lm_predictions,
    get_datasets_weights_and_num_samples
)
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

def build_dataset_group(dataset_group_name, paths, weights, splits, data_impl,
                        train_valid_test_num_samples, max_seq_length, 
                        masked_lm_prob, short_seq_prob, binary_head, seed, 
                        skip_warmup, train_valid_test):
    '''
    Build a single dataset group corresponding to Option 2 of data loading see arguments.py
    a dataset group is passed on the following form
    GIVEN_NAME WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT2 START:END PATH2
    or alternatively
    GIVEN_NAME PATH1    # for a single dataset to be used fully
    '''

    assert train_valid_test in ["train","valid","test"]

    # Single dataset.
    if len(paths) == 1:
        dataset =  _build_single_datasets(paths[0],
                                          splits[0],
                                          data_impl,
                                          train_valid_test_num_samples,
                                          max_seq_length, masked_lm_prob, 
                                          short_seq_prob, binary_head, seed, 
                                          skip_warmup, dataset_group_name, 
                                          train_valid_test)
        return dataset
    # Blending dataset.
    else:

        data_prefix = []
        # data_prefix is on the shape:
        # ["WEIGHT1", "PATH1", "WEIGHT2", "PATH2", "WEIGHT3", "PATH3"]
        for w,p in zip(weights, paths):
            data_prefix += [w,p]

        output = get_datasets_weights_and_num_samples(data_prefix,
                                                    train_valid_test_num_samples)
        prefixes, weights, datasets_train_valid_test_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_single_datasets(prefixes[i],
                                        splits[i],
                                        data_impl,
                                        datasets_train_valid_test_num_samples[i],
                                        max_seq_length,
                                        masked_lm_prob,
                                        short_seq_prob,
                                        binary_head,
                                        seed, skip_warmup,
                                        dataset_group_name, train_valid_test)

            datasets.append(ds)
        all_datasets = BlendableDataset(datasets, weights)

        return all_datasets


def _build_single_datasets(data_prefix, range_string, data_impl, train_valid_test_num_samples,
                            max_seq_length, masked_lm_prob, short_seq_prob, 
                            binary_head, seed, skip_warmup, dataset_group_name, 
                            train_valid_test):
    """Build a single dataset"""

    assert train_valid_test in ["train","valid","test"]
    index = ["train","valid","test"].index(train_valid_test)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    # this corresponds to option2 for data loading on the form
    # WEIGHT1 START:END PATH1, WEIGHT2 START:END PATH2, WEIGHT3 START:END PATH3
    # splits here is an array of size 2  [start_index, end_index]
    splits = get_split_by_range_(range_string=range_string, size=total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    print_rank_0('    {}:'.format(dataset_group_name))
    print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[0], splits[1],
                                        splits[1] - splits[0]))

    def build_dataset(name):
        dataset = None
        if splits[1] > splits[0]:
            documents = np.arange(start=splits[0], stop=splits[1],
                                  step=1, dtype=np.int32)
            dataset = BertDataset(name=name,
                                  data_prefix=data_prefix,
                                  documents=documents,
                                  indexed_dataset=indexed_dataset,
                                  max_num_samples=train_valid_test_num_samples[index],
                                  masked_lm_prob=masked_lm_prob,
                                  max_seq_length=max_seq_length, 
                                  short_seq_prob=short_seq_prob,
                                  binary_head=binary_head,
                                  seed=seed)
        return dataset

    dataset = build_dataset(dataset_group_name)

    return dataset


def get_indexed_dataset_(path, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_indexed_dataset(path,
                                           data_impl,
                                           skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


class BertDataset(torch.utils.data.Dataset):

    def __init__(self, name, data_prefix, documents, indexed_dataset, 
                 max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed, binary_head):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.binary_head = binary_head

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        # Number of tokens in each epoch and number of required epochs.
        tokens_per_epoch = _num_tokens(documents, self.indexed_dataset.sizes)
        num_epochs = _num_epochs(tokens_per_epoch, max_seq_length, max_num_samples)

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                   data_prefix,
                                                   num_epochs,
                                                   None,
                                                   self.max_seq_length - 3, # account for added tokens
                                                   short_seq_prob,
                                                   self.seed,
                                                   self.name,
                                                   self.binary_head)

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        # We % 2**32 since numpy requres the seed to be between 0 and 2**32 - 1
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
        return build_training_sample(sample, seq_length,
                                     self.max_seq_length,  # needed for padding
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, np_rng,
                                     self.binary_head)




def build_training_sample(sample,
                          target_seq_length, max_seq_length,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng, binary_head):
    """Biuld training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
    """

    if binary_head:
        # We assume that we have at least two sentences in the sample
        assert len(sample) > 1
    assert target_seq_length <= max_seq_length

    # Divide sample into two segments (A and B).
    if binary_head:
        tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample,
                                                                  np_rng)
    else:
        tokens_a = []
        for j in range(len(sample)):
            tokens_a.extend(sample[j])
        tokens_b = []
        is_next_random = False

    # Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = truncate_segments(tokens_a, tokens_b, len(tokens_a),
                                  len(tokens_b), max_num_tokens, np_rng)

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
                                                      cls_id, sep_id)

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _, _) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)

    # Padding.
    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    train_sample = {
        'text': tokens_np,
        'types': tokentypes_np,
        'labels': labels_np,
        'is_random': int(is_next_random),
        'loss_mask': loss_mask_np,
        'padding_mask': padding_mask_np,
        'truncated': int(truncated)}
    return train_sample


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np

def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence lenght, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs