# -*- coding: utf-8 -*-
"""
Created on Fri May 12 10:25:58 2017

@author: ShiCh
"""
import tensorflow as tf
import numpy as np
import reader
import pandas as pd


init_scale = 0.1        #
learning_rate = 1.0     # 学习速率
max_grad_norm = 5       # 用于控制梯度膨胀，
num_layers = 2          # lstm层数
num_steps = 30          # 单个数据中，序列的长度。
hidden_size = 200       # 隐藏层规模
max_epoch = 4           # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
max_max_epoch = 13      # 指的是整个文本循环13遍。
keep_prob = 0.8
lr_decay = 0.5          # 学习速率衰减
batch_size = 20         # 每批数据的规模，每批有20个。
vocab_size = 10000      # 词典规模，总共10K个词


raw_data = reader.ptb_raw_data("D:/simple-examples/data/") # 获取原始数据
train_data, valid_data, test_data, vocabulary= raw_data

input_data=tf.placeholder(tf.int32,[None,None])
target=tf.placeholder(tf.int32,[None,None])

epoch_size = ((len(train_data) // batch_size) - 1) // num_steps

with tf.device("/cpu:0"):    
    embedding = tf.get_variable(
                "embedding", [vocab_size, hidden_size], tf.float32) # vocab size * hidden size, 将单词转成embedding描述
            # 将输入seq用embedding表示, shape=[batch, steps, hidden_size]
    inputs = tf.nn.embedding_lookup(embedding, input_data)
                             
#train_embeded= tf.nn.embedding_lookup(embedding, input_)
#inputs = tf.nn.dropout(inputs,keep_prob)


network = tf.contrib.rnn.BasicLSTMCell(200)
output, state = tf.nn.dynamic_rnn(network, inputs, dtype=tf.float32)
output = tf.nn.dropout(output,keep_prob)

output = tf.reshape(output, [-1, 200])
#output = tf.reshape(tf.stack(axis=1, values=[output]), [-1, 200])


weight = tf.Variable(tf.truncated_normal(
                                         [200, 10000],
                                         mean=0.0,
                                         stddev=0.01,
                                         dtype=tf.float32))
bias = tf.Variable(tf.constant(0.1, shape=[10000]))




prediction = tf.matmul(output, weight) + bias


loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [prediction],
        [tf.reshape(target, [-1])],
        [tf.ones([batch_size * num_steps], tf.float32)])
#prediction = tf.reshape(prediction,[ 20,20 ])
#initial_state=network.zero_state(batch_size, tf.float32)

cost = tf.reduce_sum(loss) / batch_size
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target,logits=prediction))

global_step = tf.Variable(0, trainable=False)

initial_learning_rate = 0.01 #初始学习率

learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step=global_step,
                                           decay_steps=10,decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)

add_global = global_step.assign_add(1)

Opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
final_state=state
     
init = tf.global_variables_initializer()
sess=tf.Session()
#with tf.Session() as sess:
sess.run(init)
for step in range(25000):
    input_, targets = reader.ptb_producer(train_data, batch_size, num_steps, name="train")
    input_=np.array(input_)
    targets=np.array(targets)
    #b = np.zeros((20, 10000))
    #b[np.arange(20), targets] = 1
    cost1,state, _ ,pred= sess.run([cost, final_state, Opt,prediction], feed_dict={input_data:input_,target:targets}) # 运行session,获得cost和state
    
    if step%100==0:
        print("epoch=",step,",cost=",cost1)
        
        
test_phrase="where there is a river there is a city perhaps this is not always true but it"
test_array=test_phrase.split(" ")
vecab_table=pd.DataFrame([vocabulary])
test_vec = np.array(vecab_table[test_array])

pred= np.array(sess.run([prediction], feed_dict={input_data:test_vec})) # 运行session,获得cost和state)
pred=np.resize(pred,[20,10000])
pred=tf.nn.softmax(pred[-1,:])
vecab_index=pd.DataFrame(pred.eval(session=sess).sort_index(axis=1,ascending=False))
vocab=vecab_table.T
pred_word = vocab.ix[:,0][vocab.ix[:,0]==vecab_index]
print(pred_word.index[0])
'''
sess.close()
for epoch in range(3):
    for step in range(3):
        epoch_size = ((len(train_data) // batch_size) - 1) // num_steps
        costs = 0.0
        iters = 0
        

        cost,state, _ = sess.run([cross_entropy, final_state, Opt], feed_dict={input_data:input_.eval(session=sess),target:targets.eval(session=sess)}) # 运行session,获得cost和state


        if  step % (epoch_size // 10) == 10:  # 也就是每个epoch要输出10个perplexity值
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, cost))
'''




























                