from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from gemicai import dicomo
import os


class PickleDataFolder(IterableDataset):
    def __init__(self, base_path, dicomo_fields, transform=None):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(base_path, str), 'base_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.base_path = base_path
        self.transform = transform

    def __iter__(self):
        self.data_set_gen = self.get_next_data_set()
        self.data_set = next(self.data_set_gen)
        return self

    def __next__(self):
        try:
            while True:
                try:
                    return next(self.data_set)
                except:
                    self.data_set = next(self.data_set_gen)
        except:
            raise StopIteration

    def get_next_data_set(self):
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                yield iter(PickleDataSet(os.path.join(root, name), self.dicomo_fields, self.transform))
        raise StopIteration

class PickleDataSet(IterableDataset):

    def __init__(self, pickle_path, dicomo_fields, transform=None):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(pickle_path, str), 'pickle_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.pickle_path = pickle_path
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.pickle_stream = dicomo.stream_pickles(self.pickle_path)
            return self
        else:
            raise Exception("PickleDataSet does not support multi-process data loading")

    def __next__(self):
        try:
            # get next dicomo class from the stream
            dicomo_class = next(self.pickle_stream)

            # fetch values of the fields we are interested in
            field_list = []
            for field in self.dicomo_fields:
                try:
                    # ugly but works
                    # check if transform is specified and if it should be applied
                    temp = getattr(dicomo_class, field)
                    if self.transform and field == 'tensor':
                        try:
                            temp = self.transform(temp)
                        except:
                            raise Exception('Could not apply specified transformation to the dicom image')
                    field_list.append(temp)
                except:
                    None

            return field_list
        except:
            raise StopIteration


def print_labels_and_display_images(tensors, labels):
    for index, tensor in enumerate(tensors):
        print(labels[index])
        dicomo.plt.imshow(tensor, cmap='gray')
        dicomo.plt.show()


#Leaving it for now might be useful later
#origin = os.path.join("examples", "dicom", "CT")
#destination = os.path.join("examples", "gzip", "CT/")
#dicomo.compress_dicom_files(origin, destination, 10)

#data_directory = os.path.join("examples", "gzip", "CT")
#pickle_iterator = PickleDataFolder(data_directory, ['tensor', 'bpe'], transform=None)
#folder_iterator = torch.utils.data.DataLoader(pickle_iterator, batch_size, shuffle=False, num_workers=0)

#for tensor, bpe in pickle_iterator:
#    print(bpe)
