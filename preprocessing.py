import numpy as np

'''
    1.filter
        1.avg
    2.normalize
        1.zscore
        2.minmax
    3.channel_expand
    4.label_zoom
    5.kv_divide
'''

# filter
def filter_avg(sequences, win=3):
    # Moving average filtering
    print "filtering with win = {}.".format(win)
    result = np.asarray(sequences)
    delta = int(win/2)
    for idx, s in enumerate(sequences):
        for t in range(delta,len(s)-delta):
            result[idx][t] = np.average(s[t-delta:t+delta+1], axis=0)
    return result

# normalize
def normalize_zscore(sequences, mul=1):
    # zscore normalization
    print "normalize with zscore, multiply with {}.".format(mul)
    result = np.asarray(sequences)
    for idx, s in enumerate(sequences):
        std = np.std(s,axis=0)
        mean = np.average(s,axis=0)
        result[idx] = mul * (s-np.tile(mean,(len(s),1))) / np.tile(std, (len(s),1)).astype(np.float32)
    return result

def normalize_minmax(sequences, mul=1):
    print "normalize with minmax, multiply with {}.".format(mul)
    result = np.asarray(sequences)
    for idx, s in enumerate(sequences):
        mins = np.min(s, axis=0)
        maxs = np.max(s, axis=0)
        result[idx] = mul * (s-np.tile(mins,(len(s),1))) / np.tile(maxs-mins, (len(s),1)).astype(np.float32)
    return result

# channel expand
def channel_expand(sequences, channel=1):
    print "expand channel into {}.".format(channel)
    result = np.asarray(sequences)
    for j in range(len(sequences)):
        result[j] = np.concatenate([sequences[j][:,:,None] for _ in range(channel)],axis=-1)
    return result

# label zoom
def label_zoom(labelings, n=1):
    # assign each character n labels, i.g., zero: 0,1 one: 2,3 (n=2)
    print "zoom labeling with n = {}.".format(n)
    result = []
    for labeling in labelings:
        tmp = []
        for i in range(len(labeling)):
            tmp.extend([n*labeling[i]+j for j in range(n)])
        result.append(np.array(tmp, dtype='int64'))
    return result

# divide into k folds
def kv_divide(sequences, collectors, labels, kv_n=5):
    sequences, collectors, labels = np.asarray(sequences), np.asarray(collectors), np.asarray(labels)

    collectors_dict = {}
    for idx in range(len(sequences)):
        if not collectors_dict.has_key(collectors[idx]):
            collectors_dict[collectors[idx]] = {}
        labels_str = ''.join([str(x) for x in labels[idx]])
        if not collectors_dict[collectors[idx]].has_key(labels_str):
            collectors_dict[collectors[idx]][labels_str] = []
        collectors_dict[collectors[idx]][labels_str].append(idx)
    
    min_kv_n = min([min([len(v) for _,v in contents.iteritems()]) for _,contents in collectors_dict.iteritems()])
    kv_n = min(min_kv_n, kv_n)

    sequences_folds = [[] for _ in range(kv_n)]
    collectors_folds = [[] for _ in range(kv_n)]
    labels_folds = [[] for _ in range(kv_n)]
    for c, contents in collectors_dict.iteritems():
        for t, indexes in contents.iteritems():
            indexes_folds = np.array_split(indexes, kv_n)
            np.random.shuffle(indexes_folds)
            for fold_idx in range(kv_n):
                sequences_folds[fold_idx].extend(sequences[indexes_folds[fold_idx]])
                collectors_folds[fold_idx].extend(collectors[indexes_folds[fold_idx]])
                labels_folds[fold_idx].extend(labels[indexes_folds[fold_idx]])
    
    sequences_folds = [np.asarray(sequences_folds[i]) for i in range(kv_n)]
    collectors_folds = [np.asarray(collectors_folds[i]) for i in range(kv_n)]
    labels_folds = [np.asarray(labels_folds[i]) for i in range(kv_n)]

    return sequences_folds, collectors_folds, labels_folds