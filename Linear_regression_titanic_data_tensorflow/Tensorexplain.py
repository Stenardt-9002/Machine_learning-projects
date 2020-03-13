

import tensorflow as tf
print()
print()
print()
print()

print(tf.version)
string = tf.Variable("This is a string",tf.string)
number = tf.Variable(324,tf.int16)
rank_t = tf.Variable(["Test"],tf.string)
rank_t2 = tf.Variable([["ok","test"],["per","lol"]],tf.string)
print(rank_t)
print(rank_t2)
print(rank_t2.shape)

#reshape

tensor1 = tf.ones([1,2,3])
tesnor2 = tf.reshape(tensor1,[2,1,3])
print(tensor1)
print(tesnor2)
tesnor2 = tf.reshape(tensor1,[2,-1])
print(tesnor2)


# types
# constant
# placeholder
# varaible
# sparese tensor

#evaluate
# with tf.Session() as seess:
#     tensor.eval()



t = tf.zeros([5,5,5,5,5])
print(t)
