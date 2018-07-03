import tensorflow as tf
mod = tf.load_op_library('./libcuda_op.so')
with tf.Session() as sess:
    start = [5,4,3,2,1]
    print(start)
    print(mod.add_one(start).eval())