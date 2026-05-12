import torchvision
# This will download and extract cifar-10-python.tar.gz into ./data
torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
print('Download complete.')