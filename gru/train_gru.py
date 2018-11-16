import os
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from TextGRU import TextGRU
import time
import datetime
import logging
from sklearn.metrics import f1_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Parameters
# =====================================================================================================================
# Data loading params
tf.flags.DEFINE_string('tra_pos_file', '/Users/yangjun/CODE/PycharmProjects/datasets/kaggle_insincere_questions_classify/dataset/df_pos_tra.csv', '')
tf.flags.DEFINE_string('tra_neg_file', '/Users/yangjun/CODE/PycharmProjects/datasets/kaggle_insincere_questions_classify/dataset/df_neg_tra.csv', '')
tf.flags.DEFINE_string('val_file', '/Users/yangjun/CODE/PycharmProjects/datasets/kaggle_insincere_questions_classify/dataset/val_dataset.pkl', '')
tf.flags.DEFINE_string('embedding_file', '/Users/yangjun/CODE/PycharmProjects/datasets/kaggle_insincere_questions_classify/embeddings/embedding.pkl', '')
# Model Hyperparameters
tf.flags.DEFINE_integer("sentence_length", 25, "the max sentence length(default:12)")
tf.flags.DEFINE_integer("num_classes", 2, "the number of classes")
tf.flags.DEFINE_integer("n_neurons", 256, "the number of neurons")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.05, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate (default: 1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("total_step", 40000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("annealing_point", 3000, "the step learning rate start annealing (default: 4000)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 2000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string("log_path", './log', "file use to record train log")

# Misc Parameters
tf.flags.DEFINE_boolean("restore", False, "whether restore model")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#======================================================================================================================
#some functions will use


#learning rate schedule function, return lr acording to current step
total_steps = FLAGS.total_step
# restart_points = np.array([[0, .3], [.3, .5], [.5, .8], [.8, 1.0]])*total_steps
# def get_current_lr(step):
#     y1 = lambda x: (1e-3 - 1e-8) * x + 1e-8
#     y2 = lambda x, start, end: np.cos(((x - start) * np.pi) / (2 * (end - start)))
#     y3 = lambda x, start, end: y1(y2(x, start, end))
#     for start, end in restart_points:
#         if start < step and step <= end:
#             return y3(step, start, end)
#         else:
#             continue

#
df_pos_tra = pd.read_csv(FLAGS.tra_pos_file, engine='python')
df_neg_tra = pd.read_csv(FLAGS.tra_neg_file, engine='python')
# df_pos_val = pd.read_csv(FLAGS.val_pos_file, engine='python')
# df_neg_val = pd.read_csv(FLAGS.val_neg_file, engine='python')
#
with open(FLAGS.embedding_file, 'rb') as f:
    pretra_embed = joblib.load(f)
vocab_size, embedding_dim = pretra_embed.shape
#
with open(FLAGS.val_file, 'rb') as f:
    x_val, y_val = pickle.load(f)
#
def get_batch(batch_size, df_pos, df_neg):
    #return batch_size/2 pos examples and batch_size/2 neg exampls
    pos_batch = df_pos.sample(batch_size//2).loc[:, ['text_ind', 'target']].values
    neg_batch = df_neg.sample(batch_size//2).loc[:, ['text_ind', 'target']].values
    batch = np.vstack((pos_batch, neg_batch))
    temp = batch[:, 0]
    x = []
    for line in temp:
        line = line.replace('[', '').replace(']', '').replace('\n', '')
        line_ind = list(map(int, line.split()))
        x.append(line_ind)
    x = np.array(x)
    return x, batch[:, -1].astype(np.int64)

#======================================================================================================================
#TRAIN
def train():
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            textCly_model = TextGRU(vocab_size,
                                    embedding_dim,
                                    FLAGS.sentence_length,
                                    FLAGS.num_classes,
                                    FLAGS.n_neurons,
                                    FLAGS.l2_reg_lambda)

            train_op = textCly_model.train_op
            global_step = textCly_model.global_step

            #create a log
            current_time = '{:%Y-%m-%d_%H%M%S}'.format(datetime.datetime.now())
            log_filename = '{}.log'.format(current_time)
            logging.basicConfig(filename=FLAGS.log_path + '/' + log_filename, level=logging.INFO,
                                format="%(levelname)s %(asctime)s " +
                                       "[%(filename)s %(funcName)s %(lineno)d] %(message)s")

            logging.info('batch_size:{},num_epoches:{},annealing_point:{},l2_reg:{}'.format(
                         FLAGS.batch_size, FLAGS.total_step, FLAGS.annealing_point, FLAGS.l2_reg_lambda))

            # Output directory for models and summaries
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "gru_class_balance", current_time))
            logging.info('Writing to {}\n'.format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            #
            lr = FLAGS.learning_rate
            while True:
                #data batch
                x_tra, y_tra = get_batch(FLAGS.batch_size, df_pos_tra, df_neg_tra)
                # print(y_tra.sum())
                feed_dict = {textCly_model.input_x: x_tra,
                             textCly_model.input_y: y_tra,
                             textCly_model.dropout_keep_prob: FLAGS.dropout_keep_prob,
                             textCly_model.lr: lr,
                             textCly_model.embedding: pretra_embed}

                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, textCly_model.loss, textCly_model.accuracy],
                    feed_dict)

                time_str = datetime.datetime.now().isoformat()
                logging.info('{}: step {}, lr {}, loss {:g}, acc {:g}'.format(time_str, step, lr, loss, accuracy))
                print('{}: step {}, lr {}, loss {:g}, acc {:g}'.format(time_str, step, lr, loss, accuracy))

                #if True, lr start annealing
                # if step >= FLAGS.annealing_point:
                #     lr = get_current_lr(step)
                #validation
                if not (step+1)%FLAGS.evaluate_every:
                    #dev_step()
                    inds = np.random.choice(y_val.shape[0], 512, replace=False)
                    x_val_batch, y_val_batch = x_val[inds, :], y_val[inds]
                    # print('val {}'.format(y_val.sum()))
                    feed_dict = {textCly_model.input_x: x_val_batch,
                                 textCly_model.input_y: y_val_batch,
                                 textCly_model.dropout_keep_prob: 1.0,
                                 textCly_model.embedding: pretra_embed}

                    pre_label, loss, accuracy = sess.run([textCly_model.predictions,
                                                          textCly_model.loss,
                                                          textCly_model.accuracy],
                                                         feed_dict)

                    f1_ = f1_score(y_val_batch, pre_label)

                    time_str = datetime.datetime.now().isoformat()
                    logging.info(
                        '{}: step {}, loss {:g}, acc {:g}, f1 {}'.format(time_str, step, loss, accuracy, f1_))
                    print('{}: step {}, loss {:g}, acc {:g}, f1 {} '.format(time_str, lr, loss, accuracy, f1_))

                #save checkpoints
                if not (step+1)%FLAGS.checkpoint_every:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    logging.info('Saved model checkpoint to {}\n'.format(path))

                #whether stop
                if step >= FLAGS.total_step:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    logging.info('Saved model checkpoint to {}\n'.format(path))
                    break

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()
