import subprocess

order = 'python main.py -run {run} -network {network}'
order = order.format(run=0, network='crnn')
subprocess.call(order, shell=True)
print 'finished.'