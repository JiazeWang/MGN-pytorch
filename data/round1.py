from data.common import list_pictures

from torch.utils.data import dataset
from torchvision.datasets.folder import default_loader

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return "/mnt/SSD/jzwang/reid/ReID_data/first_round/train/"+self._data[0]

    @property
    def label(self):
        return int(self._data[1])

class Round1(dataset.Dataset):
    def __init__(self, args, transform, dtype):

        self.transform = transform
        self.loader = default_loader

        data_path = args.datadir
        self._parse_list()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.video_list[index]
        return self.get(record, segment_indices)

    def get(self, record, ):
        images = self.loader(record.path)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.imgs)
