import tensorflow as tf
import argparse
from Model import network
import time
from DataProc import load_data, make_minibatches

print("Importing done")


def train(args, test_image, test=False):

    log_file = None

    if not test:
        log_file = open(args.log_file, 'w')
        minibatchs_X, minibatchs_Y = make_minibatches(*load_data(args.datafile), args.minibatch_size)
        print("Data loaded")
        log_file.write("Data loaded\n")

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=[None, 160, 320, 3], name='X')
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

        if test:
            restore_path = tf.train.latest_checkpoint('model')
            saver.restore(sess, restore_path)
            print('restored the model from' + str(restore_path))
            pred_action = sess.run(pred, feed_dict={X: test_image})
            return pred_action

        if args.restore:
            restore_path = tf.train.latest_checkpoint(args.save_dir)
            saver.restore(sess, restore_path)
            print('restored the model from' + str(restore_path))
            log_file.write('restored the model from' + str(restore_path) + "\n")
        else:
            sess.run(init)

        summary_writer = tf.summary.FileWriter(args.summary_dir, sess.graph)

        train_start = time.time()
        print("Training started")
        log_file.write("Training started\n")

        for epoch in range(args.num_epochs):
            epoch_start = time.time()

            num_minibatches = len(minibatchs_X)
            losses = []

            for minibatch in range(num_minibatches):
                _loss, _, _summ = sess.run([loss, train_op, summary],
                                           feed_dict={X: minibatchs_X[minibatch], Y: minibatchs_Y[minibatch]})
                losses.append(_loss)

                print("Epoch " + str(epoch + 1) + " - Minibatch " + str(minibatch + 1) +
                      " completed with loss = " + str(_loss))
                log_file.write("Epoch " + str(epoch + 1) + " - Minibatch " + str(minibatch + 1) +
                               " completed with loss = " + str(_loss) + "\n")
            summary_writer.add_summary(_summ)

            print("EPOCH " + str(epoch + 1) +
                  " completed in " + str(time.time() - epoch_start)[:5] +
                  " secs with average loss = " + str(sum(losses)/len(losses)))
            log_file.write("EPOCH " + str(epoch + 1) +
                           " completed in " + str(time.time() - epoch_start)[:5] +
                           " secs with average loss = " + str(sum(losses)/len(losses)) + "\n")

            save_path = saver.save(sess, args.save_dir + 'model.ckpt')
            print("Model saved in the dir " + str(save_path))
            log_file.write("Model saved in the dir " + str(save_path) + "\n")

        print("Training Finished in " + str(time.time() - train_start)[:5])
        log_file.write("Training Finished in " + str(time.time() - train_start)[:5] + "\n")

        log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--minibatch_size', type=int, default=128)
    parser.add_argument('--restore', type=bool, default=False)
    parser.add_argument('--datafile', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--summary_dir', type=str)
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--test', type=bool, default=False)

    args = parser.parse_args()

    train(args, None)
