import argparse
import json
import logging
import os
import sys
import utils
import tensorflow as tf
import vgg19 as vgg
from style_transfer import StyleTransfer

""" Batch generator that returns one batch with one image """
class SingleImageBatchGenerator:
    def __init__(self, file_name, image_h, image_w):
        self.batch = [utils.load_image(file_name, image_h=image_h, image_w=image_w)]

    def get_batch(self):
        return self.batch

def main():
    parser = argparse.ArgumentParser()    

    parser.add_argument('--model_dir', type=str, default="output",
                        help='directory of checkpoint files')    
    parser.add_argument('--input', type=str,
                        default='data/sf.jpg',
                        help='Image to process')    
    parser.add_argument('--output', type=str, default="output.png",
                        help='exported file')
    parser.add_argument('--image_h', type=int, default=-1,
                        help='weight for texture loss vs content loss')
    parser.add_argument('--image_w', type=int, default=-1,
                        help='weight for texture loss vs content loss')

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s', 
                            level=logging.INFO,
                            datefmt='%I:%M:%S')

    with open(os.path.join(args.model_dir, 'result.json'), 'r') as f:
        result = json.load(f)

    model_name = result['model_name']
    best_model_full = result['best_model']
    best_model_arr = best_model_full.split('/')
    best_model_arr[0] = args.model_dir
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

    style = result['style']

    logging.info("Loading model from %s" % best_model)
    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope(model_name):        
            model = StyleTransfer(is_training=False, batch_size=1, style=style, 
                image_h=args.image_h, image_w=args.image_w)
        model_saver = tf.train.Saver(name='saver', sharded=True)
    try:
        with tf.Session(graph=graph) as session:
            model_saver.restore(session, best_model)
            logging.info("Processing image")
            batch_gen = SingleImageBatchGenerator(args.input, args.image_h, args.image_w)
            _, _, _, test_out, _ = model.run_epoch(session, tf.no_op(), None, batch_gen, num_iterations=1)

            utils.write_image(args.output, test_out)
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

if __name__ == '__main__':
    main()
