import numpy as np
import json

class history():
    def __init__(self):
        self.epoch = []
        self.train_ler = []
        self.val_ler = []
        self.loss = []
    def append(self, param_dict={}):
        for k,v in param_dict.iteritems():
            getattr(self, k).append(str(v))
    def save(self, despath):
        to_save = {'epoch':self.epoch, 
                   'train_ler':self.train_ler,
                   'val_ler':self.val_ler,
                   'loss':self.loss}
        with open(despath, 'wb') as f:
            json.dump(json.dumps(to_save), f)
    def print_history(self):
        for i in range(len(self.epoch)):
            log = "epoch {} train_ler={} val_ler={} loss={}"
            print log.format(self.epoch[i], self.train_ler[i], self.val_ler[i], self.loss[i])

def pad_sequences(sequences, network_len_func, maxlen=None, dtype=np.float32,
                  padding='post', truncating='post', value=0.):
    '''Pads each sequence to the same length: the length of the longest
    sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
            lengths: numpy array with the original sequence lengths
    '''
    if type(value)==str:
        value=0.
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, network_len_func(lengths)

def sparse_tuple_from(sequences, dtype=np.float32):
    '''
    Create a sparse representation of x.
    Args:
        sequences: list or asarray
    Returns:
        A tuple with (indices, values, shape)
    '''
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

def get_feed(data, targets, network, run):
    inputs, seq_lens = pad_sequences(data, network.get_conv_result_seqlen)
    targets = sparse_tuple_from(targets)

    if run=='train':
        keep_prob = 0.5
    elif run=='test':
        keep_prob = 1.0

    feed_dict = {network.inputs:inputs,
                 network.targets:targets,
                 network.seq_len:seq_lens,
                 network.keep_prob:keep_prob}
    return feed_dict
