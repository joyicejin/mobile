import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph

with tf.Graph().as_default(), tf.Session() as sess:
    # model_dir = '..\\capOut\\model\\gn+batch=1'
    model_dir = '..\\capOut\\model\\tboard\\'

    # saver = tf.train.import_meta_graph(os.path.join(model_dir, "model_fold0_max.ckpt-20.meta"))
    saver = tf.train.import_meta_graph(os.path.join(model_dir, "model_fold0_max.ckpt-49.meta"))
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    print("Model restored")

    tf.train.write_graph(sess.graph_def, model_dir, "cap49.pbtxt")

input_graph = model_dir + 'cap49.pbtxt'
# checkpoint_path = model_dir + 'model_fold0_max' + '.ckpt-20'
checkpoint_path = model_dir + 'model_fold0_max' + '.ckpt-49'
output_frozen_graph_name = 'Model' + '.pb'
output_graph = os.path.join(model_dir, output_frozen_graph_name)
input_binary = False
output_node_names = "save/restore_all"
restore_op_name = "l13/softmax"
filename_tensor_name = "save/Const:0"

# output_optimized_graph_name = 'optimized_' + '.pb'
# clear_devices = True

freeze_graph.freeze_graph(input_graph=input_graph, input_saver='', input_binary=input_binary,
                          input_checkpoint=checkpoint_path, output_node_names=output_node_names,
                          output_graph=output_graph, clear_devices=True, filename_tensor_name="save/Const:0",
                          restore_op_name="save/restore_all", initializer_nodes='')
