import argparse
import batches
import codecs
import json
import logging
import os
import shutil
import sys
import time
import utils

import numpy as np
import tensorflow as tf
import vgg19 as vgg

from batches import BatchGenerator
from generator import GeneratorNetwork
import style_transfer
from datetime import datetime

TF_VERSION = int(tf.__version__.split('.')[1])
print("Tensorflow version %d" % TF_VERSION)

tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

DEFAULT_DIR = "DEFAULT"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--zoom', type=bool, default=False,
                        help='zoom on texture a bit')
    parser.add_argument('--init_dir', type=str, default='',
                        help='restore model and continue training')

    # Directory params
    parser.add_argument('--data_dir', type=str,
                        default='input_images/',
                        help='training data dir')
    parser.add_argument('--output_dir', type=str,
                        default=DEFAULT_DIR,
                        help='output dir')
    parser.add_argument('--texture', type=str, default="./data/starry.jpg",
                        help='source texture')

    # Model params
    parser.add_argument('--model_name', type=str, default='model',
                        help='name of the model')    
    parser.add_argument('--texture_weight', type=int, default=15,
                        help='weight for texture loss vs content loss')
    parser.add_argument('--image_h', type=int, default=vgg.DEFAULT_SIZE,
                        help='weight for texture loss vs content loss')
    parser.add_argument('--image_w', type=int, default=vgg.DEFAULT_SIZE,
                        help='weight for texture loss vs content loss')
    parser.add_argument('--generator', type=str, default=style_transfer.RESIDUAL,
                        help='name of the model')    

    # Parameters to control the training.
    parser.add_argument('--batch_size', type=int, default=4,
                        help='minibatch size')
    parser.add_argument('--batch_index', type=int, default=0,
                        help='start index for images')

    parser.add_argument('--epoch_size', type=int, default=400,
                        help='iterations per epoch')    
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='number of epochs')    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--tv', type=int, default=0,
                        help='Total Variation Loss')

    # Parameters for logging.
    parser.add_argument('--log_to_file', dest='log_to_file', action='store_true',
                        help=('whether the experiment log is stored in a file under'
                              '  output_dir or printed at stdout.'))
    parser.set_defaults(log_to_file=False)

    args = parser.parse_args()

    assert args.generator == style_transfer.RESIDUAL or args.generator == style_transfer.MULTISCALE, "Only RESIDUAL and MULTISCALE models are allowed"

    if args.init_dir:
        args.output_dir = args.init_dir
    else:
        if args.output_dir == DEFAULT_DIR:
            args.output_dir = args.model_name

    # Specifying location to store model, best model and tensorboard log.
    args.save_best_model = os.path.join(args.output_dir, 'best_model/model')
    args.save_model = os.path.join(args.output_dir, 'last_model/model')
    args.tb_log_dir = os.path.join(args.output_dir, 'tensorboard_log/')
    status_file = os.path.join(args.output_dir, 'status.json')
    export_file = os.path.join(args.output_dir, 'model_exported.pb')

    best_model = None
    best_valid_loss = np.Inf

    if not args.init_dir:
        # Clear and remake paths
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        for paths in [args.save_best_model, args.save_model,
                    args.tb_log_dir, os.path.join(args.output_dir, 'images/')]:
            os.makedirs(os.path.dirname(paths))

    # Specify logging config.
    if args.log_to_file:
        args.log_file = os.path.join(args.output_dir, 'experiment_log.txt')
    else:
        args.log_file = 'stdout'

    # Set logging file.
    if args.log_file == 'stdout':
        logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s', 
                            level=logging.INFO,
                            datefmt='%I:%M:%S')
    else:
        logging.basicConfig(filename=args.log_file,
                            format='%(asctime)s %(levelname)s:%(message)s', 
                            level=logging.INFO,
                            datefmt='%I:%M:%S')        

    if args.init_dir:
        with open(os.path.join(args.init_dir, 'result.json'), 'r') as f:
            result = json.load(f)
            args.model_name = result['model_name']
            args.texture_weight = result['texture_weight']
            args.image_h = result['image_h']
            args.image_w = result['image_w']
            args.learning_rate = result['learning_rate']
            args.batch_size = result['batch_size']
            args.epoch_size = result['epoch_size']
            args.texture = result['texture']
            args.start_epoch = result['last_epoch'] + 1
            epochs = args.num_epochs
            args.num_epochs = result['num_epochs'] - args.start_epoch
            if args.num_epochs <= 0:
                args.num_epochs = epochs
            best_valid_loss = result['best_valid_loss']
            args.batch_index = result['batch_index']
            args.tv = result['tv']
    else:
        result = {}
        result['model_name'] = args.model_name
        result['texture_weight'] = args.texture_weight
        result['image_h'] = args.image_h
        result['image_w'] = args.image_w
        result['learning_rate'] = args.learning_rate
        result['batch_size'] = args.batch_size
        result['texture'] = args.texture
        result['num_epochs'] = args.num_epochs    
        result['epoch_size'] = args.epoch_size
        result['tv'] = args.tv

    logging.info("Training style: %s with texture loss weight %d" % (args.texture, args.texture_weight))


    texture = utils.load_image(args.texture, args.image_h, args.image_w, zoom=args.zoom)
    # Create graphs
    logging.info('Creating graph')
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope(args.model_name):
            train_model = style_transfer.StyleTransfer(
                is_training=True, batch_size=args.batch_size, 
                image_h=args.image_h, image_w=args.image_w, model=args.generator,
                texture=texture)
        model_saver = tf.train.Saver(name='best_model_saver', sharded=True)

    utils.write_image(os.path.join(args.output_dir,'texture.png'), [texture])

    data_dir = args.data_dir
    h = args.image_h
    w = args.image_w

    logging.info('Start session\n')
    try:
        # Use try and finally to make sure that intermediate
        # results are saved correctly so that training can
        # be continued later after interruption.
        with tf.Session(graph=graph) as session:

            #session.run(tf.initialize_all_variables())
            logging.info('Building loss function')
            with tf.name_scope('build_loss'):
                loss = train_model.build_loss(session, texture_weight=args.texture_weight, tv=args.tv)
            # Version 8 changed the api of summary writer to use
            # graph instead of graph_def.
            if TF_VERSION >= 8:
                graph_info = session.graph
            else:
                graph_info = session.graph_def

            train_writer = tf.train.SummaryWriter(args.tb_log_dir + 'train/', graph_info)
            valid_writer = tf.train.SummaryWriter(args.tb_log_dir + 'valid/', graph_info)

            logging.info('Start training')
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            train = optimizer.minimize(loss, global_step=train_model.global_step)
            last_model_saver = tf.train.Saver(name='last_model_saver', sharded=True)
            if args.init_dir:
                model_path = result['latest_model']
                last_model_saver.restore(session, model_path)
                logging.info("restored %s" % model_path)
                session.run(tf.initialize_variables([train_model.global_step]))
            else:
                session.run(tf.initialize_all_variables())

            logging.info('Get batches')
            batch_gen = BatchGenerator(args.batch_size, h, w, data_dir, max_batches=args.epoch_size, 
                                       logging=logging, batch_index=args.batch_index)
            batch_gen_valid = BatchGenerator(args.batch_size, h, w, data_dir, max_batches=args.num_epochs, valid=True)


            start_epoch = 0 if not args.init_dir else args.start_epoch
            for i in range(start_epoch, start_epoch + args.num_epochs):
                logging.info('=' * 19 + ' Epoch %d ' + '=' * 19 + '\n', i)
                logging.info('Training on training set')
                result['batch_index'] = batch_gen.get_last_load()
                # training step
                loss, train_summary_str, _, _, global_step = train_model.run_epoch(session, train, train_writer, 
                    batch_gen, num_iterations=args.epoch_size, output_dir=args.output_dir)
                logging.info('Evaluate on validation set')

                valid_loss, valid_summary_str, valid_image_summary, last_out, _ = train_model.run_epoch(session, 
                    tf.no_op(), valid_writer, batch_gen_valid, num_iterations=1, output_dir=args.output_dir)

                utils.write_image(os.path.join(args.output_dir,'images/epoch_%d.png' % i), last_out)

                saved_path = last_model_saver.save(session, args.save_model, global_step=train_model.global_step)
                logging.info('Latest model saved in %s\n', saved_path)

                # save and update best model
                if (not best_model) or (valid_loss < best_valid_loss):
                    logging.info('Logging best model')
                    best_model = model_saver.save(session, args.save_best_model)
                    best_valid_loss = valid_loss
                valid_writer.add_summary(valid_summary_str, global_step)
                valid_writer.flush()
                logging.info('Best model is saved in %s', best_model)
                logging.info('Best validation loss is %f\n', best_valid_loss)
                result['latest_model'] = saved_path
                result['last_epoch'] = i
                result['best_model'] = best_model
                # Convert to float because numpy.float is not json serializable.
                result['best_valid_loss'] = float(best_valid_loss)                
                result_path = os.path.join(args.output_dir, 'result.json')
                if os.path.exists(result_path):
                    os.remove(result_path)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2, sort_keys=True)
                save_status(i, status_file, best_valid_loss)

            # Save graph def
            tf.train.write_graph(session.graph_def, args.output_dir, "model.pb", False) 

    except:
        logging.info("Unexpected error!")
        logging.info(sys.exc_info()[0])
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        result_path = os.path.join(args.output_dir, 'result.json')
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, sort_keys=True)
        logging.info('Done!')


def save_status(epoch, status_file, ppl):
    status = {}
    status["epoch"] = epoch
    status["timestamp"] = str(datetime.now())
    status["best_valid_ppl"] = "%.4f" % ppl
    with codecs.open(status_file, 'w', encoding = 'ascii') as f:
        json.dump(status, f, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
