import numpy as np
from utils import *

record = history()
record.append({'epoch':0,'train_ler':0.99,'val_ler':0.98,'loss':100})
record.append({'epoch':10,'train_ler':0.98,'val_ler':0.97,'loss':90})
record.append({'epoch':20,'train_ler':0.97,'val_ler':0.96802163,'loss':80})

record.save('tmp.json')