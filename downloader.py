from torchvision.datasets import MNIST
import torchvision.transforms as tf

MNIST('C:\\Users\\atombob\\Desktop\\MNISTGAN\\data', download=True)


def get_training():
    return MNIST('C:\\Users\\atombob\\Desktop\\MNISTGAN\\data', train=True,
                 transform=tf.Compose([
                     tf.ToTensor(),
                     tf.Normalize((0.5, ), (0.5, )),
                 ]))


def get_test():
    return MNIST('C:\\Users\\atombob\\Desktop\\MNISTGAN\\data', train=False,
                 transform=tf.Compose([
                     tf.ToTensor(),
                     tf.Normalize((0.5, ), (0.5, )),
                 ]))
