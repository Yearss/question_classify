import tensorflow as tf


class TextGRU(object):
    """
    A RNN for text classification.
    Uses an embedding layer, followed by GRUCells and softmax layer.
    """

    def __init__(self,
                 vocab_size,
                 embed_dim,
                 sentence_length,
                 num_classes,
                 n_neurons,
                 l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, None, name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.lr = tf.placeholder(tf.float32, name="learning_rate")
        self.embedding = tf.placeholder(tf.float32, [vocab_size, embed_dim], name="embedding")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope('Embedding'):
            # static word embedding
            # with open(staticE_fp, 'rb') as f:
            #     staticE = joblib.load(f)
            # self.W_staticE = tf.get_variable('staticE',
            #                                  shape=staticE.shape,
            #                                  initializer=tf.constant_initializer(staticE),
            #                                  trainable=True)
            # a batch_size * sentence_len * embedding tensor
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # RNN
        basicGRU = tf.nn.rnn_cell.GRUCell(n_neurons)
        basicGRU = tf.nn.rnn_cell.DropoutWrapper(basicGRU, output_keep_prob=self.dropout_keep_prob)
        # init_state = basicGRU.zero_state(batch_size, tf.float32)
        # cells = tf.nn.rnn_cell.MultiRNNCell([basicGRU] * n_layers)
        # init_state = cells.zero_state(batch_size, tf.float32)
        self.output, self.final_state = tf.nn.dynamic_rnn(basicGRU,
                                                          self.embedded_chars,
                                                          dtype=tf.float32,
                                                          # initial_state=init_state,
                                                          time_major=False)

        # output
        with tf.variable_scope("output"):
            W = tf.get_variable(
                "W_out",
                shape=[n_neurons, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_out")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            #print('embedding shape:{}'.format(self.output.get_shape()))
            self.scores = tf.nn.xw_plus_b(self.final_state, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.variable_scope('train_op'):
            # Define Training procedure
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
