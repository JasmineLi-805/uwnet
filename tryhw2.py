from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 500
rate = .025
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? 
#               How does it affect convergence? How does it affect what magnitude of learning rate you can use?
#               Write down any observations from your experiments:
# 
# The loss of convnet with batch normalization decreases a lot quicker at the beginning of the training. The 
# model without batch normalization took about 200 epochs to reduce the loss to below 2.0, but with batch norm
# the model only took approx. 25 epochs. In terms of final loss and accuracy, the model with batch normalization 
# (loss=1.26, test accu=53.56%) outperforms the model without batch norm (loss=1.59, test accu=40.17%). 
# The model with batch normalization is also smoother and quicker in converging. Without batch normalization, 
# the loss oscillates a lot while decreasing, while there is a steadier decrease with batch normalization. 
# To test the effect on different learning rates, I tried lr=0.01, 0.025, 0.05, 0.075, 0.1. It turns out that 
# the best learning rate is lr=0.025, in which I received a 55.85% test accuracy. Batch normalization allows the 
# training to be more stable, thus allows larger learning rates.