import os
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Lambda
from tensorflow.keras.utils import Progbar, plot_model
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as K

K.clear_session()
from tensorflow.keras.utils import to_categorical

from model.ICR import build_feature_extractor
from model.ICR import build_classify_extractor
from model.ICR import build_domain_classify_extractor
from model.GradientReveresalLayer import GradientReversalLayer

from utils.model_utils import EarlyStopping
from utils.model_utils import grl_lambda_schedule
from utils.model_utils import learning_rate_schedule





class MNIST2MNIST_M_DANN(object):

    def __init__(self,config):

        self.cfg = config
        self.grl_lambd = 1.0  # GRL

        self.build_DANN()

        self.loss = categorical_crossentropy
        self.acc = categorical_accuracy

        self.train_loss = Mean("train_loss", dtype=tf.float32)
        self.train_cls_loss = Mean("train_cls_loss", dtype=tf.float32)
        self.train_domain_cls_loss = Mean("train_domain_cls_loss", dtype=tf.float32)
        self.train_cls_acc = Mean("train_cls_acc", dtype=tf.float32)
        self.train_domain_cls_acc = Mean("train_domain_cls_acc", dtype=tf.float32)
        self.val_loss = Mean("val_loss", dtype=tf.float32)
        self.val_cls_loss = Mean("val_cls_loss", dtype=tf.float32)
        self.val_domain_cls_loss = Mean("val_domain_cls_loss", dtype=tf.float32)
        self.val_cls_acc = Mean("val_cls_acc", dtype=tf.float32)
        self.val_domain_cls_acc = Mean("val_domain_cls_acc", dtype=tf.float32)

        #self.optimizer = tf.keras.optimizers.Adam(self.cfg.init_learning_rate)
        self.optimizer = tf.keras.optimizers.SGD(self.cfg.init_learning_rate,momentum=self.cfg.momentum_rate)
        #'''
        #self.early_stopping = EarlyStopping(min_delta=1e-5, patience=5, verbose=1)
        #'''

    def build_DANN(self):

        self.source_input = Input(shape=self.cfg.input_shape,name="input")

        self.feature_encoder = build_feature_extractor(self.cfg)
        #self._cls_encoder = build_classify_extractor((int(self.cfg.seq_len)+1,8,))
        self._cls_encoder = build_classify_extractor()
        self.domain_cls_encoder = build_domain_classify_extractor()

        self.grl = GradientReversalLayer()

        self.dann_model = Model(self.source_input,
                                [self._cls_encoder(self.feature_encoder(self.source_input)),
                                 self.domain_cls_encoder(self.grl(self.feature_encoder(self.source_input)))])
        
        self.dann_model.summary()

        if self.cfg.finetune:
            self.dann_model.load_weights(self.cfg.finetune_model_path,by_name=True,skip_mismatch=True)

    def train(self,train_source_datagen,train_target_datagen,
              val_target_datagen,train_iter_num,val_iter_num):
              
        time = self.cfg.dir_name
        checkpoint_dir = os.path.join(self.cfg.checkpoints_dir,time)
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        log_dir = os.path.join(self.cfg.logs_dir, time)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        #self.cfg.save_config(time)

        self.writer_hyperparameter = tf.summary.create_file_writer(os.path.join(log_dir,"hyperparameter"))
        self.writer_train = tf.summary.create_file_writer(os.path.join(log_dir,"train"))
        self.writer_val = tf.summary.create_file_writer(os.path.join(log_dir,'validation'))

        print('\n----------- start to train -----------\n')
        with open(os.path.join(log_dir,'log.txt'),'w') as f:
            for ep in np.arange(1,self.cfg.epoch+1,1):
                self.progbar = Progbar(train_iter_num+1)
                print('Epoch {}/{}'.format(ep, self.cfg.epoch))
                train_loss, train_cls_acc, domain_loss,domain_acc = self.train_one_epoch \
                        (train_source_datagen, train_target_datagen, train_iter_num, ep)
                val_loss, val_cls_acc = self.eval_one_epoch(val_target_datagen, val_iter_num, ep)
                self.progbar.update(train_iter_num+1, [('val_loss', val_loss),
                                                       ("val_acc", val_cls_acc)])
                self.train_loss.reset_states()
                self.train_cls_acc.reset_states()
                self.train_domain_cls_loss.reset_states()
                self.train_cls_acc.reset_states()
                self.train_domain_cls_acc.reset_states()
                self.val_loss.reset_states()
                self.val_cls_acc.reset_states()
                self.val_domain_cls_loss.reset_states()
                self.val_cls_acc.reset_states()
                self.val_domain_cls_acc.reset_states()

                # save
                str = "Epoch{:03d}-train_loss-{:.3f}-val_loss-{:.3f}-domain_loss-{:.3f}-train_cls_acc-{:.3f}-val_cls_acc-{:.3f}-domain_cls_acc-{:.3f}" \
                    .format(ep, train_loss, val_loss, domain_loss, train_cls_acc, val_cls_acc, domain_acc)
                print(str)
                f.write(str + "\n")  
                self.dann_model.save(os.path.join(checkpoint_dir, str + ".h5"))

                '''
                stop_training = self.early_stopping.on_epoch_end(ep, val_cls_acc)
                if stop_training:
                    break
                '''
        self.dann_model.save(os.path.join(checkpoint_dir, "trained_dann_mnist2mnist_m"))
        print('\n----------- end to train -----------\n')

    def train_one_epoch(self,train_source_datagen,train_target_datagen,train_iter_num,ep):
        for i in np.arange(1, train_iter_num + 1):
            batch_source_data, batch_source_labels = train_source_datagen.__next__()  # train_source_datagen.next_batch()
            batch_target_data, batch_target_labels = train_target_datagen.__next__()  # train_target_datagen.next_batch()
            a = np.array(self.cfg.batch_size//2*[0])
            b = np.array(self.cfg.batch_size//2*[1])
            batch_domain_labels = np.concatenate([a,b]).astype(np.int32)
            batch_domain_labels = to_categorical(batch_domain_labels,2)
            batch_labels = to_categorical(batch_source_labels,2)
            batch_data = np.concatenate([batch_source_data, batch_target_data], axis=0)
            # 更新学习率并可视化
            iter = (ep - 1) * train_iter_num + i
            process = iter * 1.0 / (self.cfg.epoch * train_iter_num)
            self.grl_lambd = grl_lambda_schedule(process)
            learning_rate = learning_rate_schedule(process, init_learning_rate=self.cfg.init_learning_rate)
            tf.keras.backend.set_value(self.optimizer.lr, learning_rate)
            with self.writer_hyperparameter.as_default():
                tf.summary.scalar("hyperparameter/learning_rate", tf.convert_to_tensor(learning_rate), iter)
                tf.summary.scalar("hyperparameter/grl_lambda", tf.convert_to_tensor(self.grl_lambd), iter)


            with tf.GradientTape() as tape:
                _cls_feature = self.feature_encoder(batch_source_data)
                cls_pred = self._cls_encoder(_cls_feature,training=True)
                # cls_pred = cls_pred
                _cls_loss = self.loss(batch_source_labels,cls_pred)
                _cls_acc = self.acc(batch_source_labels, cls_pred)

                if(self.cfg.Dann):
                    domain_cls_feature = self.feature_encoder(batch_data,training=True)
                    domain_cls_pred = self.domain_cls_encoder(self.grl(domain_cls_feature, self.grl_lambd),
                                                          training=True)
                    domain_cls_loss = self.loss(batch_domain_labels, domain_cls_pred)
                    domain_cls_acc = self.acc(batch_domain_labels, domain_cls_pred)
                    loss = tf.reduce_mean(_cls_loss) + tf.reduce_mean(domain_cls_loss)
                else:
                    loss = tf.reduce_mean(_cls_loss)
                a=1
            vars = tape.watched_variables()
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))

            self.train_loss(loss)
            self.train_cls_loss(_cls_loss)
            self.train_cls_acc(_cls_acc)
            if(self.cfg.Dann):
                self.train_domain_cls_acc(domain_cls_acc)
                self.train_domain_cls_loss(domain_cls_loss)
                self.progbar.update(i, [('loss', loss),
                               ('_cls_loss', _cls_loss),
                               ('domain_cls_loss', domain_cls_loss),
                               ("_acc", _cls_acc),
                               ("domain_acc", domain_cls_acc)])
            else:
                self.progbar.update(i, [('loss', loss),
                               ('_cls_loss', _cls_loss),
                               ("_acc", _cls_acc)])
        # 可视化损失与指标
        with self.writer_train.as_default():
            #print(os.getcwd())
            tf.summary.scalar("loss/loss", self.train_loss.result(), ep)
            tf.summary.scalar("loss/_cls_loss", self.train_cls_loss.result(), ep)
            tf.summary.scalar("acc/_cls_acc", self.train_cls_acc.result(), ep)
            if(self.cfg.Dann):
                tf.summary.scalar("acc/domain_cls_acc", self.train_domain_cls_acc.result(), ep)
                tf.summary.scalar("loss/domain_cls_loss", self.train_domain_cls_loss.result(), ep)

        return self.train_loss.result(),self.train_cls_acc.result(),self.train_domain_cls_loss.result(),self.train_domain_cls_acc.result()


    def eval_one_epoch(self, val_target_datagen, val_iter_num, ep):
        for i in np.arange(1, val_iter_num + 1):
            batch_target_data, batch_target_labels = val_target_datagen.__next__()
            target_feature = self.feature_encoder(batch_target_data)
            target_cls_pred = self._cls_encoder(target_feature, training=False)
            target_cls_loss = self.loss(batch_target_labels, target_cls_pred)
            _cls_acc = self.acc(batch_target_labels, target_cls_pred)
            if(self.cfg.Dann):
                batch_target_domain_labels = np.array((self.cfg.batch_size) * [1])            
                batch_target_domain_labels = to_categorical(batch_target_domain_labels, 2)
                target_domain_cls_pred = self.domain_cls_encoder(target_feature, training=False)
                target_domain_cls_loss = self.loss(batch_target_domain_labels, target_domain_cls_pred)
                target_loss = tf.reduce_mean(target_cls_loss) + tf.reduce_mean(target_domain_cls_loss)
                domain_cls_acc = self.acc(batch_target_domain_labels, target_domain_cls_pred)
            else:
                target_loss = tf.reduce_mean(target_cls_loss)
            self.val_loss(target_loss)
            self.val_cls_loss(target_cls_loss)
            self.val_cls_acc(_cls_acc)
            if(self.cfg.Dann):
                self.val_domain_cls_loss(target_domain_cls_loss)
                self.val_domain_cls_acc(domain_cls_acc)            
                self.progbar.update(i, [('val_loss', target_loss),
                               ('val_cls_loss', target_cls_loss),
                               ('val_domain_cls_loss', target_domain_cls_loss),
                               ("val_cls_acc", _cls_acc),
                               ("domain_acc", domain_cls_acc)])
            else:
                self.progbar.update(i, [('val_loss', target_loss),
                               ('val_cls_loss', target_cls_loss),
                               ("val_cls_acc", _cls_acc)])
        # 可视化验证损失及其指标
        with self.writer_val.as_default():
            tf.summary.scalar("loss/loss", self.val_loss.result(), ep)
            tf.summary.scalar("loss/_cls_loss", self.val_cls_loss.result(), ep)
            tf.summary.scalar("acc/_cls_acc", self.val_cls_acc.result(), ep)
            if(self.cfg.Dann):
                tf.summary.scalar("loss/domain_cls_loss", self.val_domain_cls_loss.result(), ep)
                tf.summary.scalar("acc/domain_cls_acc", self.val_domain_cls_acc.result(), ep)
        return self.val_loss.result(), self.val_cls_acc.result()

    
