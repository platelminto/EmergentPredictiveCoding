import argparse
import mnist
import Network
import random
import torch
import os
from functions import get_device
from train import train
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='device')
parser.add_argument('--i', type=str, help='Device index')
args = parser.parse_args()

DEVICE = get_device()

INPUT_SIZE = 28*28
BATCH_SIZE = 32
SEQ_LENGTH = 10

train_set, validation_set, test_set = mnist.load(val_ratio=0.0)

os.makedirs("models/patterns_rev/seeded_mnist", exist_ok=True)

"""
Create and train ten instances of energy efficient RNNs for MNIST 
"""
n_instances = 1  # number of model instances
# losses = [str(beta)+'beta'+'l1_postandl2_weights' for beta in [3708.0] ]
losses = ['l1_pre', 'l1_post', [str(beta) + 'beta' + 'l1_postandl2_weights' for beta in [3708.0]][0]]
# losses = ['l1_pre']
# seeds = [[random.randint(0,10000) for i in range(n_instances)], \
seeds = [[random.randint(0,10000) for i in range(n_instances)] for j in range(len(losses))]

# train MNIST networks
for loss_ind, loss in enumerate(losses):
    for i in range(0, n_instances):
        print("loss", loss_ind, "instance", i)
        net = Network.State(activation_func=torch.nn.ReLU(),
                            optimizer=torch.optim.Adam,
                            lr=1e-4,
                            input_size=INPUT_SIZE,
                            hidden_size=INPUT_SIZE,
                            title="patterns_rev/seeded_mnist/mnist_net_" + loss+"_"+str(i),
                            device=DEVICE,
                            seed=seeds[loss_ind][i])
        train(net,
              train_ds=train_set,
              test_ds=test_set,
              loss_fn=loss,
              num_epochs=100,
              batch_size=BATCH_SIZE,
              sequence_length=SEQ_LENGTH,
              verbose=False)

        # plot test and train loss from net.results dict (keys: 'train loss', 'test loss')
        plt.plot(net.results['train loss'], label='train loss')
        plt.plot(net.results['test loss'], label='test loss')
        plt.legend()
        plt.title('Train and Test Loss for '+loss)
        plt.show()

        # save model
        net.save()

# """
# Create and train ten instances of energy efficient RNNs for CIFAR10
# """
# INPUT_SIZE = 3072
# HIDDEN_SIZE = 3072 # add 32 to this number if you want to have extra latent resources
# BATCH_SIZE = 32
# SEQ_LENGTH = 10
# LOSS_FN = 'l1_pre'
#
# training_set, validation_set, test_set = cifar.load(val_ratio=0.0, color=True)
#
# """
# Create and train ten instances of energy efficient RNNs with cifar 10
# # """
# N = 10 # number of model instances per seed
#
# seeds = [random.randint(0,10000) for i in range(N)]
#
# for i in range(N):
#
#         cifar_net= Network.State(activation_func=torch.nn.ReLU(),
#         optimizer=torch.optim.Adam,
#         lr=1e-4,
#         input_size=INPUT_SIZE,
#         hidden_size=HIDDEN_SIZE,
#         title="/final_networks/seeded_cifar_nets/cifar_net_"+str(i),
#         device=DEVICE,
#         seed=seeds[i])
#
#
#         cifar_net.save()
#
#         train(cifar_net,
#               train_ds=training_set,
#               test_ds=test_set,
#               loss_fn=LOSS_FN,
#               num_epochs=1000,
#               batch_size=BATCH_SIZE,
#               sequence_length=SEQ_LENGTH,
#               verbose=False
#               )
#         ## save model
#         cifar_net.save()
