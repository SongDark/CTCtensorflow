from preprocessing import *

'''
    *.npy: [data, collectors, labelings]
        data: [[Txd],...]
        collectors: [str,...]
        labelings: [str,...]
'''

def datapreparation(params):
    
    data, collectors, labelings = np.load(params.data_path)
    normalize_func = {'zscore':normalize_zscore, 'minmax':normalize_minmax}[params.norm]
    
    # data = filter_avg(data, win=params.filtering_win)
    data = normalize_func(data, params.norm_mul)
    data = channel_expand(data, channel=params.expand_channel)

    vocabulary = sorted(list(set(''.join(labelings))))
    params.modify({"labels":vocabulary, "num_classes":params.label_zoom*len(vocabulary)+1})

    labelings = [[params.labels.index(c) for c in labeling] for labeling in labelings]
    
    labelings = label_zoom(labelings, n=params.label_zoom)

    data, collectors, labelings = kv_divide(data, collectors, labelings, kv_n=params.kv_n)
    
    return data, labelings