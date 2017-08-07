import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)



def conv_layer(input, filter, kernel, stride = (1,1), layer_name="conv") :
    with tf.name_scope(layer_name) :
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

class DenseNet() :
    def __init__(self, x, nb_blocks, filters) :
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.model = self.build_model(x)

    def bottleneck_layer(self, x, scope) :
        # print(x)
        with tf.name_scope(scope) :
            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')

            x = tf.layers.batch_normalization(x)
            x = tf.nn.relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')


            # print(x)
            return x

    def transition_layer(self, x, scope) :
        with tf.name_scope(scope) :
            x = tf.layers.batch_normalization(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = tf.layers.average_pooling2d(inputs=x, pool_size=2, strides=2, padding='SAME')

            return x

    def dense_block(self, input_x, nb_layers, layer_name) :
        with tf.name_scope(layer_name) :
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers) :
                # print(i)
                x = tf.concat(layers_concat, axis=3)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i+1))
                layers_concat.append(x)

            return x


    def build_model(self, input_x) :
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], layer_name='conv0')
        x = tf.layers.max_pooling2d(inputs=x, pool_size=3, strides=2, padding='SAME')

        for i in range(self.nb_blocks) :
            # print(i)

            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_final')
        x = tf.layers.dense(inputs=x, units=10, name='fully_connected')

        x = tf.reshape(x, [-1,10])
        return x


growth_k = 12
batch_size = 100


x = tf.placeholder(tf.float32, shape=[None, 784])
batch_images = tf.reshape(x, [-1, 28, 28, 1])

y_ = tf.placeholder(tf.float32, shape=[None, 10])


logits = DenseNet(x=batch_images, nb_blocks=5, filters=growth_k).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
train = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())
training_epochs = 20

with tf.Session() as sess :
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) :
        saver.restore(sess, ckpt.model_checkpoint_path)
    else :
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    global_step = 0
    for epoch in range(training_epochs) :
        total_batch = int(mnist.train.num_examples / batch_size)

        for step in range(total_batch) :
            batch_x, batch_y = mnist.train.next_batch(batch_size)

            feed_dict = {
                x : batch_x,
                y_ : batch_y
            }



            _, loss= sess.run([train,cost], feed_dict=feed_dict)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # writer.add_summary(graph, global_step=step)



            if step % 100 == 0 :
                global_step += 100
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=feed_dict)
                    # accuracy.eval(feed_dict=feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer.add_summary(train_summary, global_step=global_step)

            test_feed_dict = {
                x : mnist.test.images,
                y_ : mnist.test.labels
            }


        accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
        print('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates)
        # writer.add_summary(test_summary, global_step=epoch)

    saver.save(sess=sess, save_path='./model/dense.ckpt')