import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as opt
import torch.utils.data as udata
#  import discordhook
import arch
import MNISTDataset
import torchvision.utils as vutils
import torchvision.transforms.functional as fun
import PIL.Image as pil
import os


def init_weights(net):
    type_name = net.__class__.__name__
    if type_name.find('Conv') != -1 or type_name.find('Linear') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif type_name.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)


if __name__ == '__main__':

    discriminator = nn.DataParallel(arch.Disc())
    generator = nn.DataParallel(arch.Gene())
    discriminator.apply(init_weights)
    generator.apply(init_weights)

    opt_disc = opt.Adam(discriminator.parameters(), lr=0.0000146)
    opt_gene = opt.Adam(generator.parameters(), lr=0.0000146)

    batch_size = 20000

    training_data = data.DataLoader(MNISTDataset.MNIST(), batch_size=batch_size, num_workers=12, shuffle=True)
    loss = nn.BCELoss()
    dev = torch.device('cuda')

    loaded = torch.load('save.pt')
    discriminator.load_state_dict(loaded['disc'])
    generator.load_state_dict(loaded['gene'])

    opt_disc.load_state_dict(loaded['opt_disc'])
    opt_gene.load_state_dict(loaded['opt_gene'])

    for epoch in range(1000):

        for item, data in enumerate(training_data, 0):

            discriminator.zero_grad()

            true_out = discriminator(data[0])
            err_true = loss(true_out, torch.full((batch_size, 1,), 0, dtype=torch.float32, device=dev))
            err_true.backward()

            false = generator(torch.randn((batch_size, 1, 64), dtype=torch.float32, device=dev))
            false_out = discriminator(false.detach())
            err_false = loss(false_out, torch.full((batch_size, 1,), 1, dtype=torch.float32, device=dev))
            err_false.backward()
            disc_loss = err_true.item() + err_false.item()

            opt_disc.step()
            generator.zero_grad()

            gene_out = discriminator(false)
            err_gene = loss(gene_out, torch.full((batch_size, 1,), 0, dtype=torch.float32, device=dev))
            err_gene.backward()
            gene_loss = 2 * err_gene.item()

            opt_gene.step()

            progress = '[{}/{}][{}/{}]'.format(epoch, 1000, item, 60000 / batch_size)
            full_message = '{:<30}Discriminator Error True: ' \
                           '{:<22.10}Discriminator Error False: {:<22.10}Generator Error: {:<25.10}'.format(
                            progress, err_true, err_false, err_gene)
            print(full_message)
            vutils.save_image(fun.to_tensor(fun.resize(fun.to_pil_image(false[0].detach().cpu()),
                                                       [280, 280], pil.NONE)), 'sample.png', normalize=True)
        torch.save({
            'disc': discriminator.state_dict(),
            'gene': generator.state_dict(),
            'opt_disc': opt_disc.state_dict(),
            'opt_gene': opt_gene.state_dict()
        }, 'save.pt')
        torch.save({
            'disc': discriminator.state_dict(),
            'gene': generator.state_dict(),
            'opt_disc': opt_disc.state_dict(),
            'opt_gene': opt_gene.state_dict()
        }, 'back.pt')
