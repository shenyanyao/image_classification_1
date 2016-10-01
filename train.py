from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from datetime import datetime
import time
from six.moves import xrange
import sys, os
import re
from util import *
from yearbook_lib import *

THIS_DIRECTORY = path.dirname(path.abspath(__file__))
BATCH_SIZE = 1
MAX_STEPS = 10000


# Load all the training files 
#yb = listYearbook(True, False) # this is a list of dirs

# Get the labels
import numpy as np
#years = np.array([label(y) for y in yb])

import tensorflow as tf
sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess, coord=coord)

def train():
	with tf.Graph().as_default():
		print("start training")
		global_step = tf.Variable(-1, trainable=False)
		images, labels = yearbook_input(False, BATCH_SIZE)
		logits = yearbook_inference(images)
		loss = yearbook_loss(logits, labels)
		train_op = yearbook_train(loss, global_step)
		saver = tf.train.Saver(tf.all_variables())
		#summary_op = tf.merge_all_summaries()
		init = tf.initialize_all_variables()
		sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
		sess.run(init)
		
		summary_writer = tf.train.SummaryWriter('./summary/', sess.graph)
		ckpt = tf.train.get_checkpoint_state('./result')
		if ckpt and ckpt.model_checkpoint_path:
			print("Continue training from the model {}".format(ckpt.model_checkpoint_path))
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("Train from Scratch")
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
		while not coord.should_stop():
			start_time = time.time()
			#print(labels.eval(session=sess))
			_, loss_value, step, logits_out, labels_out = sess.run([train_op, loss, global_step, logits, labels])
			print (labels_out)
			#print(step)
			duration = time.time() - start_time
			assert not np.isnan(loss_value)
			if step%1 == 0:
				num_examples_per_step = BATCH_SIZE
				examples_per_sec = num_examples_per_step / duration
				sec_per_batch = float(duration)
				format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
				print (format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
			#if step % 100 == 0:
			#	summary_str = sess.run(summary_op)
			#	summary_writer.add_summary(summary_str, step)
			if step % 100 == 0 or (step + 1) == MAX_STEPS:
				checkpoint_path = os.path.join('./result/', 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step = step)

# Train
#med = np.median(years, axis=0)
train()
# Save the model
#open(path.join(THIS_DIRECTORY,'model.txt'),'w').write('%d\n'%med)
