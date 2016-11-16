import argparse
import logging
import os
import sys
import utils
import tensorflow as tf

from datetime import datetime

def main():
    parser = argparse.ArgumentParser()    
 
    parser.add_argument('--model_file', type=str, default='data/vg-30.pb',
                        help='Pretrained model file to run')

    parser.add_argument('--input', type=str,
                        default='data/sf.jpg',
                        help='Input image to process')    
    parser.add_argument('--output', type=str, default="output.png",
                        help='Output image file')

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout,
                            format='%(asctime)s %(levelname)s:%(message)s', 
                            level=logging.INFO,
                            datefmt='%I:%M:%S')

    with open(args.model_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
        graph = tf.get_default_graph()
    
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as session:
        graph_info = session.graph

        logging.info("Initializing graph")
        session.run(tf.initialize_all_variables())
    
        model_name = os.path.split(args.model_file)[-1][:-3]            
        image = graph.get_tensor_by_name("import/%s/image_in:0" % model_name)
        out = graph.get_tensor_by_name("import/%s/output:0" % model_name)

        shape = image.get_shape().as_list()
        target = [utils.load_image(args.input, image_h=shape[1], image_w=shape[2])]
        logging.info("Processing image")
        start_time = datetime.now()
        processed = session.run(out, feed_dict={image: target})
        logging.info("Processing took %f" % ((datetime.now()-start_time).total_seconds()))
        utils.write_image(args.output, processed)
        logging.info("Done")


if __name__ == '__main__':
    main()
