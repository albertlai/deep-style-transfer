import argparse
import freeze_graph
import json
import logging
import os
import shutil
import sys
import utils

import tensorflow as tf
import vgg19 as vgg

from batches import BatchGenerator
from style_transfer import StyleTransfer

DEFAULT_MODEL = "DEFAULT"
def main():
    parser = argparse.ArgumentParser()    

    parser.add_argument('--input_dir', type=str, default="output",
                        help='directory of checkpoint files')    
    parser.add_argument('--output', type=str, default=DEFAULT_MODEL,
                        help='exported file')
    parser.add_argument('--image_h', type=int, default=-1,
                        help='weight for texture loss vs content loss')
    parser.add_argument('--image_w', type=int, default=-1,
                        help='weight for texture loss vs content loss')

    parser.add_argument('--noise', type=float, default=0.,
                        help='noise magnitude')

    logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s', 
                            level=logging.INFO,
                            datefmt='%I:%M:%S')
    
    args = parser.parse_args()
    tmp_dir = os.path.join(args.input_dir, 'tmp')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    ckpt_dir = os.path.join(tmp_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    args.save_model = os.path.join(ckpt_dir, 'model')

    with open(os.path.join(args.input_dir, 'result.json'), 'r') as f:
        result = json.load(f)

    model_name = result['model_name']
    best_model_full = result['best_model']
    best_model_arr = best_model_full.split('/')
    best_model_arr[0] = args.input_dir
    best_model = os.path.join(*best_model_arr)

    if args.image_w < 0:
        if 'image_w' in result:
            args.image_w = result['image_w']
        else:
            args.image_w = vgg.DEFAULT_SIZE
    if args.image_h < 0:
        if 'image_h' in result:
            args.image_h = result['image_h']
        else:
            args.image_h = vgg.DEFAULT_SIZE


    if args.output == DEFAULT_MODEL:
        args.output = model_name + ".pb"

    logging.info("loading best model from %s" % best_model)

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope(model_name):        
            model = StyleTransfer(is_training=False, batch_size=1, 
                image_h=args.image_h, image_w=args.image_w, inf_noise=args.noise)
        model_saver = tf.train.Saver(name='saver', sharded=True)
    try:
        with tf.Session(graph=graph) as session:

            logging.info("Loading model")
            model_saver.restore(session, best_model)

            logging.info("Verify model")
            batch_gen_valid = BatchGenerator(1, args.image_h, args.image_w, valid=True)
            _, _, _, test_out, _ = model.run_epoch(session, tf.no_op(), None, batch_gen_valid, num_iterations=1)

            utils.write_image(os.path.join(args.input_dir,'export_verify.png'), test_out)

            logging.info("Exporting model")
            best_model = model_saver.save(session, args.save_model)
            # Save graph def
            tf.train.write_graph(session.graph_def, tmp_dir, "temp_model.pb", False) 

            saver_def = model_saver.as_saver_def()
            input_graph_path = os.path.join(tmp_dir, "temp_model.pb")
            input_saver_def_path = "" # we dont have this
            input_binary = True
            input_checkpoint_path = args.save_model
            output_node_names = model_name + "/output" 
            restore_op_name = saver_def.restore_op_name 
            filename_tensor_name = saver_def.filename_tensor_name
            output_graph_path = os.path.join(args.input_dir, args.output)
            clear_devices = False

            freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                      input_binary, input_checkpoint_path,
                                      output_node_names, restore_op_name,
                                      filename_tensor_name, output_graph_path,
                                      clear_devices, None)
            shutil.rmtree(tmp_dir)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

if __name__ == '__main__':
    main()
