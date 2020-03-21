import tensorflow as tf
from numpy import random

import tensorflow as tf
from numpy import random

# set how much curves do you want to show in a graph
writer_1 = tf.summary.FileWriter("./Logs/plot1")
writer_2 = tf.summary.FileWriter("./Logs/plot2")
writer_3 = tf.summary.FileWriter("./Logs/plot3")
writer_4 = tf.summary.FileWriter("./Logs/plot4")

log_var = tf.Variable(0.0)
tf.summary.scalar("discounted_reward", log_var)

write_op = tf.summary.merge_all()

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

for i in range(100000): # set here into training timestep you used
    # for writer 1
    summary = session.run(write_op, {log_var: random.rand()})
    writer_1.add_summary(summary, i)
    writer_1.flush()

    # for writer 2
    summary = session.run(write_op, {log_var: random.rand()})
    writer_2.add_summary(summary, i)
    writer_2.flush()

    # for writer 3
    summary = session.run(write_op, {log_var: random.rand()})
    writer_3.add_summary(summary, i)
    writer_3.flush()

    # for writer 4
    summary = session.run(write_op, {log_var: random.rand()})
    writer_4.add_summary(summary, i)
    writer_4.flush()

# python plot_tensorboard.py
# tensorboard --logdir ./Logs
