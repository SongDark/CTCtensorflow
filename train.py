import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
from utils import *

def train(params, network, data, targets):
    print 'training.'

    for val_id in range(params.kv_n if params.kv_n>0 else 1):
        # construct training-set and validation-set
        train_folds_indexes = range(params.kv_n)
        train_folds_indexes.remove(val_id)
        data_val, targets_val = data[val_id], targets[val_id]
        data_train = np.concatenate([data[train_id] for train_id in train_folds_indexes], axis=0)
        targets_train = np.concatenate([targets[train_id] for train_id in train_folds_indexes], axis=0)
        print "fold {} train size = {}, val size = {}".format(val_id+1, len(data_train), len(data_val))
        params.modify({'num_samples':len(data_train)})

        # Main codes
        with tf.Session(graph=network.graph) as session:
            tf.global_variables_initializer().run()

            saver = tf.train.Saver()
            if len(params.load_model) != 0:
                print "load model from {}".format(params.load_model)
                saver.restore(session, os.path.join(params.save_path, params.load_model))
            
            start = time.time()
            record = history()

            for curr_epoch in range(params.num_epochs):
                batch_cost, train_cost, train_ler = 0., 0., 0.

                num_batches_per_epoch = int(params.num_samples / params.batch_size)
                indexes = [range(i*params.batch_size, (i+1)*params.batch_size) for i in range(num_batches_per_epoch)]
                if num_batches_per_epoch * params.batch_size < params.num_samples:
                    indexes.append(range(num_batches_per_epoch*params.batch_size, params.num_samples))

                for batch_index in indexes:      
                    # print 'batch_size =',len(batch_index)        
                    train_feed = get_feed(data_train[batch_index], targets_train[batch_index], network, "train")
                    if (curr_epoch+1)%params.show_every_epoch == 0:
                        # it would be quite slow
                        batch_cost, _, loss, ler = session.run([network.cost, network.optimizer, network.loss, network.ler], feed_dict=train_feed)
                        train_ler += ler * len(batch_index)
                        train_cost += batch_cost * len(batch_index)
                    else:
                        # it would be quite slow
                        _, loss = session.run([network.optimizer, network.loss], feed_dict=train_feed)
                        
                # validation
                if (curr_epoch+1)%params.show_every_epoch == 0:
                   val_feed = get_feed(data_val, targets_val, network, 'test')
                   val_ler = session.run(network.ler, feed_dict=val_feed)

                if (curr_epoch+1)%params.show_every_epoch == 0:
                    train_cost /= params.num_samples
                    train_ler /= params.num_samples
                    log = "fold {} Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, val_ler = {:.3f}, total_loss = {:.3f}, time = {:.3f}"
                    print log.format(val_id+1, curr_epoch+1, params.num_epochs, train_cost, train_ler, val_ler, sum(loss), time.time() - start)
                    if train_ler <= 0.001 or val_ler <=0.001:
                        break
                    record.append({'epoch':curr_epoch+1,
                                   'train_ler':train_ler,
                                   'val_ler':val_ler,
                                   'loss':sum(loss)})
                else:
                    print "fold {} Epoch {}/{} loss = {:.3f} time = {:.3f}".format(val_id+1, curr_epoch+1, params.num_epochs, sum(loss), time.time() - start)

                if params.save_every_epoch > 0:
                    if (curr_epoch+1)%params.save_every_epoch == 0:
                        record.save(os.path.join(params.save_path, "history.json"))       
                        if len(params.save_model)!=0:
                            saver.save(session, os.path.join(params.save_path, params.save_model))
            
            # save model
            record.save(os.path.join(params.save_path, "history.json")) 
            if len(params.save_model) != 0:
                saver.save(session, os.path.join(params.save_path, params.save_model))
            
            