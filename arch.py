import torch.nn as nn
import torch

do_bias = False
relu_factor = 0.2
yes = 512


class Disc(nn.Module):

    def __init__(self):
        super(Disc, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(784, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, 1),
            nn.Sigmoid()
        )
        '''
        self.y1 = nn.Sequential(
            nn.Conv2d(1, yes, 4, 2, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
        )

        self.y2 = nn.Sequential(
            nn.Conv2d(yes, yes, 3, 2, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
        )

        self.y3 = nn.Sequential(
            nn.Conv2d(yes, yes, 4, 2, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
        )

        self.y4 = nn.Sequential(
            nn.Conv2d(yes, 1, 5, 1, 0, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.Sigmoid()
        )

        '''
        '''
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, 2, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(32)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, 2, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(64)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(64, 128, 7, 1, 3, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 7, 1, 0, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(128),
        )

        self.flatten = nn.Sequential(
            nn.Conv1d(1, 1, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(128, 32),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(32, 8),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        '''

    def forward(self, x):

        x = self.seq(torch.flatten(x, 1))
        '''
        x = self.y1(x)
        x = self.y2(x)
        x = self.y3(x)
        x = self.y4(x)
        x = torch.flatten(x, 1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = torch.flatten(x, 1)
        x = torch.unsqueeze(x, 1)
        x = self.flatten(x)
        '''
        return x


class Gene(nn.Module):

    def __init__(self):
        super(Gene, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(64, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, yes),
            nn.Dropout(0.3),
            nn.LeakyReLU(relu_factor, True),
            nn.Linear(yes, 784),
            nn.Tanh()
        )

        '''
        self.the_rest = nn.Sequential(
            nn.ConvTranspose2d(1, yes, 7),
            nn.LeakyReLU(relu_factor, True),
            nn.Conv2d(yes, yes, 5, 1, 2),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.ConvTranspose2d(yes, yes, 8),
            nn.LeakyReLU(relu_factor, True),
            nn.Conv2d(yes, yes, 5, 1, 2),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.ConvTranspose2d(yes, 1, 8),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

        
        self.t1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, yes, 3, 1, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.Conv2d(yes, yes, 3, 1, 1, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
        )

        self.t2 = nn.Sequential(
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.ConvTranspose2d(yes, yes, 5, 3, 2),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
        )

        self.t3 = nn.Sequential(
            nn.Conv2d(yes, yes, 5, 1, 3, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.ConvTranspose2d(yes, yes, 7, 1, 2),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
        )

        self.t4 = nn.Sequential(
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.Conv2d(yes, yes, 5, 1, 2, bias=do_bias),
            nn.LeakyReLU(relu_factor, True),
            nn.BatchNorm2d(yes),
            nn.ConvTranspose2d(yes, 1, 7, 1, 2),
            nn.Tanh()
        )
        '''

    def forward(self, x):
        return torch.reshape(self.seq(x), (20000, 1, 28, 28))
