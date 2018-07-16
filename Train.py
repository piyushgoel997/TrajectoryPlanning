import tensorflow as tf
import argparse
from Model import network
import time
from DataProc import load_data, make_minibatches


def train(args):
    minibatchs_X, minibatchs_Y = make_minibatches(load_data(args.datafile), args.minibatch_size)

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=[None, 320, 162, 3], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 4], name='Y')

    pred = network(X)

    with tf.name_scope('train'):
        loss = tf.reduce_mean(tf.pow(Y - pred, 2))
        tf.summary.scalar('loss', loss)
        train_op = tf.train.AdamOptimizer().minimize(loss)

    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.save_dir)
            saver.restore(sess, restore_path)
            print('restored the model from' + str(restore_path))
        else:
            sess.run(init)

        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        train_start = time.time()
        print("Training started")

        for epoch in range(args.num_epochs):
            epoch_start = time.time()

            _loss, _, _summ = sess.run([loss, train_op, summary],
                                       feed_dict={X: minibatchs_X[epoch], Y: minibatchs_Y[epoch]})

            summary_writer.add_summary(_summ)

            print("Epoch " + str(epoch + 1) +
                  " completed in " + str(time.time() - epoch_start)[:5] +
                  " secs with loss " + str(_loss))

            if (epoch + 1) % 100 == 0:
                save_path = saver.save(sess, args.save_dir + 'model.ckpt')
                print("Model saved in the dir " + str(save_path))

        print("Training Finished in " + str(time.time() - train_start)[:5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO change the number of epochs, batch size - if rqd.
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--minibatch_size', type=int, default=2)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--datafile', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--summary_dir', type=str)

    args = parser.parse_args()

    train(args)
