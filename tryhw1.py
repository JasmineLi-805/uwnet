from uwnet import *

USE_CONV = True

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def fc_net():
    l = [   make_connected_layer(3072, 328),
            make_activation_layer(RELU),
            make_connected_layer(328, 192),
            make_activation_layer(RELU),
            make_connected_layer(192, 127),
            make_activation_layer(RELU),
            make_connected_layer(127, 78),
            make_activation_layer(RELU),
            make_connected_layer(78, 36),
            make_activation_layer(RELU),
            make_connected_layer(36, 17),
            make_activation_layer(RELU),
            make_connected_layer(17, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

if USE_CONV:
    m = conv_net()
else:
    m = fc_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train)) # 70.45% (conv)
print("test accuracy:     %f", accuracy_net(m, test))  # 65.59% (conv)

# ------------------------------------------------------------------------------------ #
# Q1: How many operations does the convnet use during a forward pass?

# The number of operations in the convnet: 1108480 
# 	        width	height	in_c	out_c	kernel	stride		# of operations
# Conv2d	32	    32	    3	    8	    3	    1		        221184
# Conv2d    16	    16	    8	    16	    3	    1	    	    294912
# Conv2d    8	    8	    16	    32	    3	    1   		    294912
# Conv2d    4	    4	    32	    64  	3   	1   		    294912
# FC	    2	    2	    64	    10	    -	    -		        2560

# total 						                                    1108480

# * Only matmul() operations are counted

# ------------------------------------------------------------------------------------ #
# Q2: How accurate is the fully connected network vs the convnet when they use similar 
# number of operations?

#           train accurary      test accuracy
# FC net        46.35%              44.17%
# Conv net      70.45%              65.69%

# The convolutional network works better than the fully connected layers when using
# similar number of operations.

# ------------------------------------------------------------------------------------ #
# Q3: Why are you seeing these results? Speculate based on the information you've 
# gathered and what you know about DL and ML.

# The convolutional network utilizes the spatial locality in the image data by 
# applying filters on a small portion of the data at a time. This reduces the amount 
# operations compared to a fully connected layer. Thus when the number of operations 
# is the same, the convolutional network is able to extract more features.
# Also if we look at the input and output in the intermediate layers, the data in 
# the convolutional layers have a larger spatial size compared to the corresponding 
# layer in the fully connected layer. This indicates that the convolutional network 
# is able to preserve more information from the original data, which could be useful
# for the later classification task. 

