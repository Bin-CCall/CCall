import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation,Dropout
from tensorflow.keras.layers import Input, Embedding, GRU, Lambda,Bidirectional, BatchNormalization,LayerNormalization,Convolution1D, MaxPooling1D, Attention
import tensorflow.keras.backend as K
# from lstm_ln import LSTM_LN
K.clear_session()
from model.AttentionWithContext import AttentionWithContext
from tensorflow.keras.models import Sequential, load_model, Model
import numpy as np

def build_feature_extractor(cfg):
    params = {}
    params['bit_num'] = 260
    params['emb_dim'] = 8
    
    main_input = Input(cfg.input_shape) 
    if(cfg.embed):
        model_1 = load_model(cfg.embed_model_path)
        weights_1 = model_1.get_weights()
        token_embed = Embedding(params['bit_num'], params['emb_dim'], input_length=int(cfg.seq_len)+1,weights = [np.array(weights_1[1])],name='word2vec')(main_input)
    else:
        token_embed = Embedding(params['bit_num'], params['emb_dim'], input_length=int(cfg.seq_len)+1,name='word2vec')(main_input)
    #maxlen = tf.shape(main_input)[-1]
    maxlen = cfg.seq_len+1
    #assert(maxlen==cfg.seq_len or maxlen==cfg.seq_len+1)
    print('**************************************')
    print((main_input.shape,cfg.seq_len,maxlen))
    pos_list = [0,1,2,3]
    positions = [pos_list[i%4] for i in range(maxlen)]
    positions = tf.reshape(positions,[1,-1])
    position_embed = Embedding(input_dim=maxlen, output_dim=params['emb_dim'],name='word2vec_1')(positions)
    #embed = token_embed + position_embed
    embed = token_embed
    print("the final shape of embed is: {shape}".format(shape = tf.shape(embed)))
    embed = Dropout(0.25)(embed)
    

    hidden1=Dense(32, activation='relu')(embed)
    hidden2=Dense(32, activation='relu')(hidden1)
    hidden3=Dense(16, activation='relu')(hidden2)
    hidden4=Dense(16, activation='relu')(hidden3)
    # embeddings_model = Model(inputs=main_input, outputs=[embed])
    embeddings_model = Model(inputs=main_input, outputs=[hidden4])

    return embeddings_model
    
def build_classify_extractor():
    params = {}
    params['h_dim']= 128
    params['o_dim'] = 2
    options = {}
    options['filters'] = 128
    options['filter_len'] = 4
    options['pool'] = 3
    model = tf.keras.Sequential([Convolution1D(filters=options['filters'],
                                kernel_size=options['filter_len'],
                                strides = options['filter_len'],
                                padding='valid'),
                                Dropout(0.25),
                                #LayerNormalization(),
                                BatchNormalization(),
                                Activation('relu'),
                                Bidirectional(GRU(params['h_dim'], return_sequences=True)),
                                Bidirectional(GRU(params['h_dim'], return_sequences=True)),
                                # Bidirectional(LSTM_LN(params['h_dim'], return_sequences=True)),
                                # Bidirectional(LSTM_LN(params['h_dim'], return_sequences=True)),
                                #BatchNormalization(),
                                Dropout(0.25),
                                AttentionWithContext(),                        
                                Dense(params['o_dim'],name="cls_pred"),
                                #LayerNormalization(),
                                BatchNormalization(),
                                #BatchNormalization(),
                                Activation('softmax')

    ])
    return model

def build_domain_classify_extractor():
    params = {}
    params['h_dim']= 64
    model = tf.keras.Sequential([GRU(params['h_dim'], return_sequences=True),
                                 GRU(params['h_dim'], return_sequences=False),
                                 BatchNormalization(),
                                 Activation('relu'),
                                 #tf.keras.layers.Dropout(0.5),
                                 Dense(2, activation='softmax', name="domain_cls_pred")
    ])
    return model
