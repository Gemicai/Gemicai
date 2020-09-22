from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
import os

from gemicai import dicomo


class PickledDicomoDataFolder(IterableDataset):
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
                print(name)
                yield iter(PickledDicomoDataSet(os.path.join(root, name), self.dicomo_fields, self.transform))
        raise StopIteration


class PickledDicomoDataSet(IterableDataset):

    def __init__(self, pickle_path, dicomo_fields, transform=None):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(pickle_path, str), 'pickle_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.pickle_path = pickle_path
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.pickle_stream = self.stream_pickled_dicomos()
            return self
        else:
            raise Exception("PickledDicomoDataSet does not support multi-process data loading")

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

    def stream_pickled_dicomos(self):
        tmp = dicomo.tempfile.NamedTemporaryFile(mode="ab+", delete=False)
        try:
            dicomo.unzip_to_file(tmp, self.pickle_path)
            while True:
                yield dicomo.pickle.load(tmp)
        except EOFError:
            pass
        finally:
            tmp.close()
            os.remove(tmp.name)

def print_labels_and_display_images(tensors, labels):
    for index, tensor in enumerate(tensors):
        print(labels[index])
        dicomo.plt.imshow(tensor, cmap='gray')
        dicomo.plt.show()
