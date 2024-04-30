import os
import pickle

import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from config.config import config
from utils.dataset_utils import batch_generator,split_dataset
from model.MNIST2MNIST_M_train import MNIST2MNIST_M_DANN


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def run_main():
    seq_len = 1664
    portion = 1
    seqencetonum = True
    Dann = False
    finetune = False
    #input_shape = (seq_len,)
    init_learning_rate = 3e-2
    momentum_rate = 0.9
    batch_size = 64
    #epoch = 200
    
    finetune_model_path = os.path.join(os.path.abspath(os.getcwd()),"model_trained","<model_name>")

    embed_model_path = os.path.abspath(os.getcwd())+"/embedding/CNN_w2v_C.h5"
    checkpoints_dir = os.path.abspath(os.getcwd())+"/checkpoints/models/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size)
    if(os.path.exists(checkpoints_dir)):
        print('yes')
    logs_dir = os.path.abspath(os.getcwd())+"/logs/new/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size)
    config_dir = os.path.abspath(os.getcwd())+"/config/models/SGD-lr={0}-momentum={1}/batch_size={2}".format(init_learning_rate,momentum_rate,batch_size)
    
    source_dataset_path = "./data/C_train_positive.pkl"
    target_dataset_path = "./data/C_test_positive.pkl" # for test set
    
    cfg = config(embed_model_path = embed_model_path,
                 finetune_model_path = finetune_model_path,
                 finetune=finetune,
                 dir_name =os.path.abspath(os.path.abspath(os.getcwd())+"/model_nodann"),
                 checkpoints_dir = checkpoints_dir,
                 logs_dir = logs_dir,
                 config_dir = config_dir,
                 input_shape = (int(seq_len)+1,),
                 seq_len = seq_len,
                 init_learning_rate = init_learning_rate,
                 momentum_rate= momentum_rate,
                 batch_size=batch_size,
                 epoch = 10,
                 portion = portion,
                 Dann = Dann
                 )

    print('data split...')

    if(seqencetonum):
        x_train,y_train,x_val,y_val = split_dataset(source_dataset_path,cfg.seq_len,cfg.portion,0)
        target_x_train, target_y_train, target_x_val, target_y_val = split_dataset(target_dataset_path,cfg.seq_len,cfg.portion,0)
    else:
        source = pkl.load(open(source_dataset_path, 'rb'))
        x_train=np.array(source['0'][:int(0.9*len(source['0']))])
        x_val=np.array(source['0'][int(0.9*len(source['0'])):])
        y_train = np.array(source['1'][:int(0.9*len(source['1']))])
        y_val = np.array(source['1'][int(0.9*len(source['1'])):])
       

        # Load target
        target = pkl.load(open(target_dataset_path, 'rb'))
        target_x_train = target['0'][:int(0.9*len(target['0']))]
        target_x_val = target['0'][int(0.9*len(target['0'])):]

    #print(np.array([x_train, to_categorical(y_train)]).shape)
    print('.....',len(x_train),'.....')
    train_source_datagen = batch_generator([x_train, to_categorical(y_train)],cfg.batch_size // 2)
    train_target_datagen = batch_generator([target_x_train, to_categorical(target_y_train)],cfg.batch_size // 2)
    val_target_datagen = batch_generator([target_x_val, to_categorical(target_y_val)],cfg.batch_size)

    train_source_batch_num = int(len(x_train)/(cfg.batch_size//2))
    train_target_batch_num = int(len(target_x_train)/(cfg.batch_size//2))
    train_iter_num = int(np.max([train_source_batch_num,train_target_batch_num]))
    val_iter_num = int(len(x_val)/cfg.batch_size)

    model = MNIST2MNIST_M_DANN(cfg)
    model.train(train_source_datagen,train_target_datagen,val_target_datagen,train_iter_num,val_iter_num)




if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_main()

