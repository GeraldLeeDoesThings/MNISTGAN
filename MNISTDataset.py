from torch.utils.data import Dataset
import downloader


class MNIST(Dataset):

    def __init__(self):
        self.data = downloader.get_training()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
