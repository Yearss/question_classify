
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sklearn.metrics as metrics
from sklearn.externals import joblib
import tensorflow as tf
import nltk
import os 
import re
from tensorflow.python import debug as tf_debug


pd.set_option("display.max_colwidth",1000)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# In[2]:


train = pd.read_csv("../input/train_.csv")
dev = pd.read_csv("../input/dev.csv")
test = pd.read_csv("../input/test.csv")


# In[3]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"’","'", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


# In[4]:


train["question_word"] = list(map(clean_text, train["question_text"]))
dev["question_word"] = list(map(clean_text, dev["question_text"]))
test["question_word"] = list(map(clean_text, test["question_text"]))


# In[5]:


epoch = 3
batch_size = 256
max_length = 70
mem_dim = 200
embed_dim = 300
proj_dim = 200
vocab_size = 100000
lr = 0.005
display_step = 100
eval_step = 10000
dmodel = 500
dropout = 1
atten_dim = 200
att_len = 8


# In[6]:


w2tf = {}
for words in train["question_word"]:
    for word in words.split(" "):
        if word not in w2tf:
            w2tf[word] = 0
        w2tf[word] += 1
w_tf = sorted(w2tf.items(), key=lambda x:x[1], reverse=True)
w2idx = {w[0]:idx for w, idx in zip(w_tf[:vocab_size], range(2,vocab_size))}
w2idx["PAD"] = 0
w2idx["UNK"] = 1


# In[7]:


def load_vec(filename):
        
        mu, sigma = 0, 1 / vocab_size
        matrix_embed = np.random.normal(mu, sigma, [vocab_size, embed_dim])
        cnt = 0
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                word, *n = line.strip().split(" ")
                if word in w2idx:
                    cnt += 1
                    vector = np.array(n, np.float32)
                    matrix_embed[w2idx[word],:] = vector
                    
        oov_rate = (vocab_size - cnt) / vocab_size
        return matrix_embed, oov_rate
pretrained_emb, oov_rate = load_vec("../input/glove.6B.300d.txt")


# In[8]:


def sent2idsAndPad(sent, max_len):
    ids = [w2idx[w] if w in w2idx else 1 for w in sent.split(" ")]
    n = len(ids)
    if n >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - n)


# In[9]:


tf.reset_default_graph()
with tf.variable_scope("input"):
    question_input = tf.placeholder(name="question_text", shape=[None, max_length], dtype=tf.int32)
    question_len = tf.placeholder(name="question_mask", shape=[None], dtype=tf.int32)
    target_input = tf.placeholder(name="target", shape=[None], dtype=tf.int32)

with tf.variable_scope("embedding"):
    embeding = tf.get_variable(name="embedding", initializer=tf.constant(value=pretrained_emb, dtype=tf.float32))
    # Model 
    question_embed = tf.nn.embedding_lookup(ids=question_input, params=embeding)
    # queustion_embed_drop = tf.nn.dropout(x=question_embed, keep_prob=dropout)

with tf.variable_scope("lstm_attention"):
    fc = tf.contrib.rnn.GRUCell(num_units=mem_dim)
    fca = tf.contrib.rnn.AttentionCellWrapper(attn_length=att_len, cell=fc)
    outputs, _ = tf.nn.dynamic_rnn(cell=fca, sequence_length=question_len, dtype=tf.float32, inputs=question_embed)
    print(outputs.shape)
    # final_out = tf.nn.max_pool(value=tf.expand_dims(outputs, axis=-1), ksize=[1,max_length, 1, 1], strides=[1,1,1,1], padding="VALID")
    final_out = tf.reduce_max(outputs, axis=1, keep_dims=False)
    print(final_out.shape)
# den = tf.squeeze(input=final_out, axis=[1,3])
# # den = tf.layers.dense(inputs=rnn_hidden_drop, activation=tf.nn.relu, units=dmodel)
# # den_drop = tf.nn.dropout(x=den, keep_prob=dropout)
# # y_hat = tf.squeeze(tf.layers.dense(inputs=den, activation=tf.nn.sigmoid, units=1), axis=1)
# print(den.shape)
# y_hat = tf.squeeze(tf.contrib.layers.fully_connected(activation_fn=tf.sigmoid, inputs=den, num_outputs=1), axis=1)
with tf.variable_scope("output"):
    w = tf.get_variable("w", shape=[mem_dim, 1], dtype=tf.float32)
    b = tf.get_variable("b", shape=[], dtype=tf.float32)
    aff = tf.matmul(final_out, w) + b
    y_hat = tf.squeeze(tf.sigmoid(aff), axis=1)
    print(y_hat.shape)

with tf.variable_scope("loss"):
    y = tf.cast(target_input, dtype=tf.float32)
    loss = -tf.reduce_mean(y*tf.log(y_hat)+(1-y)*tf.log(1-y_hat))

with tf.variable_scope("optim"):
    optimer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    _, auc = tf.metrics.auc(labels=target_input, predictions=y_hat)

with tf.variable_scope("summary"):
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var.value())
        tf.summary.histogram("prob", y_hat)
        tf.summary.histogram("embed",embeding)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("auc", auc)
        tf.summary.histogram("y_hat", y_hat)
        # tf.summary.histogram("den", den)
        tf.summary.histogram("w", w)
        tf.summary.histogram("aff", aff)
        tf.summary.histogram("final_out", final_out)
        tf.summary.histogram("question_eb", question_embed)
        


# In[13]:


with tf.Session() as sess:
    sess = tf_debug.TensorBoardDebugWrapperSession(sess, "100.88.66.8:8080")      
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    summary_op = tf.summary.merge_all()
    fw = tf.summary.FileWriter("./log/run6")
    cnt = 0
    for epch in range(epoch):
        j = 0
        while j < len(train):
            batch_x = [sent2idsAndPad(q, max_length) for q in train["question_word"][j:j+batch_size]]
            batch_y = [target for target in train.target[j:j+batch_size]]

            _, cost, hat, auc_ =  sess.run([optimer, loss, y_hat, auc], feed_dict={
                question_input:batch_x,
                target_input:batch_y,
                question_len: [len(q.split(" ")) for q in train["question_word"][j:j+batch_size]]
            })
    #         print("den avg:{0}, var:{1}".format(np.mean(den_), np.var(den_)))
    #         print("w:{0}, var:{1}".format(np.mean(w_), np.var(w_)))
    #         print("aff:{0}, var:{1}".format(np.mean(aff_), np.var(aff_)))
    #         print("hat, avg:{0}, var:{1}".format(np.mean(hat), np.var(hat)))
            if cnt % display_step == 0: 
                print("epoch:{0},iter:{1}, loss:{2}, acc:{3}".format(epch, cnt, cost, auc_))
    #             print(hat)
                fw.add_summary(sess.run(summary_op, feed_dict={
                                    question_input:batch_x,
                                    target_input:batch_y,
                                    question_len: [len(q.split(" ")) for q in train["question_word"][j:j+batch_size]]
                                }), cnt)

            cnt += 1
            j += batch_size
        best_thre = evaluate(train.sample(n=10000), sess)
        evaluate(dev, sess)


# In[ ]:


def evaluate(val, sess):
    
    def search_threshold(thresholds, proba):
        max_f1, max_thre = 0., 0.
        for threshold in thresholds:
            val["prediction"] = np.asarray(proba > threshold, np.int64)
            f1 = metrics.f1_score(y_pred=val["prediction"], y_true=val["target"])
            if max_f1 < f1:
                max_f1, max_thre = f1, threshold

        return max_f1, max_thre
        
    probs = sess.run(y_hat, feed_dict={
            question_input:[sent2idsAndPad(q, max_length) for q in val["question_word"]],
            target_input:[target for target in val.target],
            question_len: [len(q.split(" ")) for q in val["question_word"]]
        })
   
    fpr, tpr, thresholds = metrics.roc_curve(val["target"], probs)
    _, opt_thre = search_threshold(thresholds, probs)
    val["prediction"] = np.asarray(probs > opt_thre, np.int64)
    cm = confusion_matrix(y_pred=val["prediction"], y_true=val["target"])
    print(cm)
    print(metrics.f1_score(y_pred=val["prediction"], y_true=val["target"]))
    print(metrics.accuracy_score(y_pred=val["prediction"], y_true=val["target"]))
    print(classification_report(y_pred=val["prediction"], y_true=val["target"]))
    print(opt_thre)
    return opt_thre
    






