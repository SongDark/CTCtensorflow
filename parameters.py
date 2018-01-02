# coding:utf-8
class parameters():  
    def __init__(self):
        
        # data parameters
        self.data_path = ""
        self.num_samples = -1
        self.num_features = 6
        self.num_classes = -1
        self.num_channels = 1
        self.labels = []

        # pre-processing parameters
        self.filtering_win = 3
        self.norm = 'minmax' # minmax or zscore
        self.norm_mul = 255.0 # only used in minmax
        self.expand_channel = 1
        self.label_zoom = 2

        # k-folds validation
        self.kv_n = 5

        # # network type
        # self.network_type = 'crnn'

        # training hyper-parameters
        self.num_epochs = 10000
        self.batch_size = -1
        self.show_every_epoch = 25
        self.draw_every_epoch = 50
        self.save_every_epoch = 200
        self.save_path = None
        self.load_model = ''
        self.save_model = ''

    
    def modify(self, param_dict={}):
        # modify parameters
        if type(param_dict) == str:
            # allow input a path
            with open(param_dict,'rb') as fi:
                param_dict = eval(fi.read())
        for k, v in param_dict.iteritems():
            if hasattr(self, k):
                if getattr(self, k) != v:
                    setattr(self, k, v)
            else:
                raise ValueError('%s is not defined in parameters.' % k)

    def print_params(self, args=[]):
        # print parameters
        if len(args)==0:
            args = dir(self)
        for attr in dir(self):
            if attr in args:
                if not (str(type(getattr(self, attr)))[7:-2] == 'instancemethod' or attr[:2] == "__"):
                    print "{} = {}".format(attr, getattr(self, attr))
        print
