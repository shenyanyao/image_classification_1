from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from six.moves import xrange
import sys, os
import re
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import numpy as np
import tensorflow as tf
from os import path
import model1
from util import *
TOWER_NAME = 'tower'
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL  = 5009
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 400
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

NUM_CLASSES = 109
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.
THIS_DIRECTORY = path.dirname(path.abspath(__file__))
BATCH_SIZE = 1
MAX_STEPS = 10000
# yearbook_activation_summary:
# yerabook_read_image:
#

'''
# for training
def yearbook_distorted_inputs(): # however, we did not distort the inputs
	return yearbook_input(False, BATCH_SIZE)
# for testing
def yearbook_inputs():
	return yearbook_input(True, BATCH_SIZE)
'''

def _yearbook_variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape, initializer=initializer)
	return var 
def _yearbook_variable_with_weight_decay(name, shape, stddev, wd):
	var = _yearbook_variable_on_cpu(name, shape,
		tf.truncated_normal_initializer(stddev=stddev))
	if wd is not None:
		weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var
def yearbook_train(total_loss, global_step):
	#num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
	#decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

	lr = 1e-3
	opt = tf.train.AdamOptimizer(learning_rate=lr)
	train_op = opt.minimize(total_loss, global_step)
	'''
	tf.scalar_summary('learning_rate', lr)
	loss_averages_op = _yearbook_add_loss_summaries(total_loss)
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.AdamOptimizer(lr)
		grads = opt.compute_gradients(total_loss)
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	#for var in tf.trainable_variables():
	#	tf.histogram_summary(var.op.name, var)
	#for grad, var in grads:
	#	if grad is not None:
	#		tf.histogram_summary(var.op.name + '/gradients', grad)
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())
	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')
	'''
	return train_op

'''
def _yearbook_add_loss_summaries(total_loss):
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses+[total_loss])
	for l in losses + [total_loss]:
		tf.scalar_summary(l.op.name + ' (raw)', l)
		tf.scalar_summary(l.op.name, loss_averages.average(l))
	return loss_averages_op
'''


def yearbook_inference(images):
	data_path = '/home/yanyao/Dropbox/Graduate Course/1. Deep Learning Seminar/CS395T-project1/src/imagenet-vgg-verydeep-19.mat'
  	vgg_net, mean_pixel = model1.net(data_path, images)
  	relu7 = vgg_net['relu7']
  	# local3
  	with tf.variable_scope('lastlayer') as scope:
    	# Move everything into depth so we can perform a single matrix multiply.
		reshape = tf.reshape(relu7, [-1, 4096])
    	dim = reshape.get_shape()[1].value
    	# 
    	
    	weights = _yearbook_variable_with_weight_decay('weights', shape=[dim, NUM_CLASSES],
                                          stddev=0.04, wd=0)
    	biases = _yearbook_variable_on_cpu('biases3', [NUM_CLASSES], tf.constant_initializer(0.1))
    	softmax_linear = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
	return softmax_linear

    	# if another choice:
    	'''
    	weights = _yearbook_variable_with_weight_decay('weights', shape=[dim, 1],
                                          stddev=0.04, wd=0.004)
    	biases = _yearbook_variable_on_cpu('biases3', [1], tf.constant_initializer(0.1))
    	linear = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
	return linear
	'''

def yearbook_loss(logits, labels):
	'''
	return tf.reduce_sum(tf.abs(tf.sub(logits, tf.cast(labels, tf.float32))))
	'''
	labels = tf.cast(labels, tf.int64)
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels, name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
	return cross_entropy_mean
	#tf.add_to_collection('losses', cross_entropy_mean)
	#return tf.add_n(tf.get_collection('losses'), name='total_loss')

def yearbook_loss_l1(logits, labels):
	labels = tf.cast(labels, tf.int64)
	logits = tf.argmax(logits)
	return tf.reduce_mean(tf.abs(tf.sub(labels, logits)))

'''
def yearbook_activation_summary(x):
	tensor_name = re.sub('%s_[0-82]*/' % TOWER_NAME, '', x.op.name) #???
	tf.histogram_summary(tensor_name + '/activations', x)
	tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
'''

def yearbook_read_image(filename_queue):
	# input.label
	# input.uint8image
	label = tf.sub(filename_queue[1], tf.constant(1905))
	#print(label)
	file_contents = tf.read_file('../data/yearbook/'+filename_queue[0])
	image = tf.image.decode_png(file_contents, channels=3)
	return image, label 
	'''
	class YearBookRecord(object):
		pass
	result = YearBookRecord()
	reader = tf.TFRecordReader()
	_, serialized = reader.read(filename_queue)
	
	features = tf.parse_single_example(serialized, features = {
    	'label': tf.FixedLenFeature([], tf.string),
    	'image': tf.FixedLenFeature([], tf.string),
    	})
	record_image = tf.decode_raw(features['image'], tf.uint8)
	result.image = tf.reshape(record_image, [224,224,1])
	result.label = tf.cast(features['label'], tf.string)
	result.label = tf.sub(tf.string_to_number(result.label, tf.int32), tf.constant(1905))

	return result
	'''


def _yearbook_generate_image_and_label_batch(image, label, 
				 min_queue_examples, batch_size, shuffle):
	# Create a queue that shuffles the examples, and then
	# read 'batch_size' images + labels from the example queue.
	#num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			#num_threads=num_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size,
			min_after_dequeue=min_queue_examples
			)
	else:
		images, label_batch = tf.train.batch(
			[image, label],
			batch_size=batch_size,
			#num_threads=num_preprocess_threads,
			capacity= 3 * batch_size
			)

	# Display the training images in the visualizer.
	#tf.image_summary('images', images)

	return images, tf.reshape(label_batch, [batch_size])

training_dataset_i, training_dataset_l = listYearbook(True, False) 
testing_dataset_i, testing_dataset_l  = listYearbook(False, True) 

def yearbook_input(eval_data, batch_size):
	'''
	if not eval_data:
		#filename = "../data/yearbook_bin/training-images/training-image-*.tfrecords"
		filename_list = training_dataset
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		#filename = "../data/yearbook_bin/testing-images/testing-image-*.tfrecords"
		filename_list = testing_dataset
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
	'''
	#_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename))
	if not eval_data:
		filename_queue = tf.train.slice_input_producer([training_dataset_i, training_dataset_l], shuffle=True)
	else:
		filename_queue = tf.train.slice_input_producer([testing_dataset_i, testing_dataset_l], shuffle=True)


	#read_input = yearbook_read_image(_filename_queue)
	read_input_image, read_input_label = yearbook_read_image(filename_queue)
	image = tf.image.convert_image_dtype(read_input_image, tf.float32)
	grayscale_image = tf.image.rgb_to_grayscale(image)
	resized_image = tf.image.resize_images(grayscale_image, 224, 224)
	float_image = tf.image.per_image_whitening(resized_image)

	return tf.train.batch([float_image, read_input_label], batch_size=batch_size, capacity=3*batch_size)

	'''
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)
	return _yearbook_generate_image_and_label_batch(float_image, 
				read_input.label, min_queue_examples, batch_size,
				shuffle=False)
	'''
