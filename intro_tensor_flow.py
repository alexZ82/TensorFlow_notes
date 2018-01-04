import tensorflow as tf
import numpy as np
import random


graph = tf.Graph() #create a Graph object
with graph.as_default():
    with tf.name_scope('variables'): #variables for summaries
        global_step = tf.Variable(0,name='global_step',trainable = False, dtype=tf.int32) #step counter
        total_output = tf.Variable(0,trainable = False, dtype= tf.float32, name ='total_output')# accumulator of output
    with tf.name_scope('transformation'):
        with tf.name_scope('input_layer'):
            a = tf.placeholder(dtype=tf.float32,shape=[None],name = 'input_placeholder_a')#input layer using a placeholder
        with tf.name_scope('first_layer'):
            b = tf.reduce_prod(a,name = 'product_b')#node that multiplies inputs
            c = tf.reduce_sum(a,name ='sum_c')#node that adds inputs
        with tf.name_scope('output'):
            output = tf.add(b,c,name = 'output')# output layer, adds intermediate nodes
    with tf.name_scope('update'):
        increment_step = global_step.assign_add(1)#incrementing global_step variable
        update_total = total_output.assign_add(output) #summing output
    with tf.name_scope('summaries'):
        avg = tf.div(update_total, tf.cast(increment_step,tf.float32),name = 'average_output')
        tf.summary.scalar('total_summary',update_total)
        tf.summary.scalar('average_summary',avg)
    with tf.name_scope('global_ops'):
        init = tf.global_variables_initializer()#initialise variables
        merge_summaries = tf.summary.merge_all()

sess = tf.Session(graph=graph)
sess.run(init)
LOGDIR = './newlog'
writer = tf.summary.FileWriter(LOGDIR)
writer.add_graph(graph)
def run_graph(input_tensor):
    input_dict = {a:input_tensor}
    _,step,summary = sess.run([output,increment_step,merge_summaries],feed_dict=input_dict)
    writer.add_summary(summary,global_step=step)

run_graph([1,2,3])
run_graph([4,5])
for i in range(10):
    run_graph(np.random.rand(random.randint(2,8)))


writer.flush()
writer.close()
sess.close()
