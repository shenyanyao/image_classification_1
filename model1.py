import tensorflow as tf 
import numpy as np 
import scipy.io 

def net(data_path, input_image):
	layers = [
		'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

		'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

		'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
		'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

		'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
		'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

		'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
		'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
	]
	data = scipy.io.loadmat(data_path)
	mean = data['normalization'][0][0][0]    	# ?
	mean_pixel = np.mean(mean, axis = (0,1)) 	# ?
	weights = data['layers'][0] 				# ?
	net = {}
	current = preprocess(input_image, mean_pixel)
	kernels, bias = weights[0][0][0][0][0]
	for i in xrange(37):
		name = layers[i]
		kind = name[:4]
		#print i
		#print kind
		if kind == 'conv':
			#print i
			kernels, bias = weights[i][0][0][0][0]
			# in matconvnet, weights are [width, height, inchannels, outchannels]
			# tensorflow: weights are [height, width, inchannels, outchannels]
			kernels = np.transpose(kernels, (1,0,2,3))
			bias = bias.reshape(-1)
			current = _conv_layer(current, kernels, bias)
		elif kind == 'relu':
			current = tf.nn.relu(current)
		elif kind == 'pool':
			current = _pool_layer(current)
		net[name] = current
	assert len(net) == len(layers)
	# 'fc6' & 'relu6'
	current = tf.reshape(current, [-1, 7*7*512])
	kernels, bias = weights[37][0][0][0][0]
	kernels = np.transpose(kernels, (1,0,2,3))
	kernels = tf.reshape(kernels, [-1, 4096])
	current = tf.matmul(current, kernels)+bias
	net['fc6'] = current
	current = tf.nn.relu(current)
	net['relu6'] = current
	# 'fc7'
	# 'relu7'
	kernels, bias = weights[39][0][0][0][0]
	kernels = tf.reshape(kernels, [-1, 4096])
	current = tf.matmul(current, kernels)+bias
	net['fc7'] = current
	current = tf.nn.relu(current)
	net['relu7'] = current
	# 'fc8'
	# 'prob'

	return net, mean_pixel

def _conv_layer(input, weights, bias):
	conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1,1,1,1), padding='SAME')
	return tf.nn.bias_add(conv, bias)

def _pool_layer(input):
	return tf.nn.max_pool(input, ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')	

def preprocess(image, mean_pixel):
	return image-mean_pixel

def unprocess(image,mean_pixel):
	return image+mean_pixel