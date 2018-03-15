import tensorflow as tf
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

state = tf.Variable(5, name='fuck')

print(state.name)
one = tf.constant(10)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(3):
        #sess.run(update)
        print(sess.run(state))
