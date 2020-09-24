import gemicai.data_iterators as iterators
from gemicai.dicomo import LabelCounter
from torchsummary import summary
from datetime import datetime
import torch.nn as nn
import torchvision
import pickle
import torch
import os


class Classifier:
    def __init__(self, base_model=nn.Module, loss_function=None, optimizer=None, verbosity_level=0, enable_cuda=False, cuda_device=None):
        # Sets base model of the classifier
        if not isinstance(base_model, nn.Module):
            raise Exception("base_model should have a base class of nn.Module")
        self.model = base_model

        # select a correct cuda device
        if enable_cuda:
            if not torch.cuda.is_available():
                raise Exception("cuda is not available on this machine")

            device_name = "cuda"
            if cuda_device is not None:
                if not isinstance(cuda_device, int) or cuda_device < 0:
                    raise Exception("cuda_device parameter should be eiter set to None or be a non-negative integer")
                device_name += ":" + str(cuda_device)

            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cpu")

        # set a proper loss function
        if loss_function is None:
            self.loss_function = nn.CrossEntropyLoss()
        elif not isinstance(loss_function, nn.CrossEntropyLossImpl):
            raise Exception("Custom loss_function should have a base class of nn.CrossEntropyLossImpl")
        else:
            self.loss_function = loss_function

        # set a proper optimizer
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise Exception("Custom optimizer should have a base class of torch.optim.Optimizer")
        else:
            self.optimizer = optimizer

        if not isinstance(verbosity_level, int):
            raise Exception("verbosity_level parameter should be of an integer type")
        self.verbosity_level = verbosity_level

        # TODO IS THE REST REALLY NEEDED?
        # Data loader's metadata , used to calculate benchmarks.
        self.dl_train_directory = None
        # self.dl_total_images = None

        # Input shape of the tensors used by the classifier, only needed for keras like model summary
        self.input_shape = (3, 244, 244)

    # TODO REFACTOR
    def set_trainable_layers(self, layers, boolean):
        for name, param in self.model.named_parameters():
            name = '.'.join(name.split('.')[:-1])
            if layers == 'all' or name in layers:
                param.requires_grad = boolean

    # TODO REFACTOR
    def summary(self):
        summary(self.model, self.input_shape)

    def evaluate(self, data_set=None, batch_size=4, num_workers=0):
        Classifier.validate_data_set_parameters(data_set=data_set, batch_size=batch_size, num_workers=num_workers)

        correct, total = 0, 0
        if not data_set.can_be_parallelized():
            num_workers = 0
        data_loader = torch.utils.data.DataLoader(data_set, batch_size, shuffle=False, num_workers=num_workers)

        classes = self.determine_classes(data_loader)
        is_pinned = data_set.is_pinned()

        # puts model in evaluation mode.
        self.model.eval()
        self.model = self.model.to(self.device)

        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images = images.to(self.device)
                labels = torch.tensor([classes.index(label) for label in labels]).to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Total: {} -- Correct: {} -- Accuracy: {}%'.format(total, correct, round(100 * correct / total, 2)))

    def train(self, data_set=None, batch_size=4, epochs=20, num_workers=0):
        Classifier.validate_data_set_parameters(data_set, batch_size, epochs, num_workers)

        if not data_set.can_be_parallelized():
            num_workers = 0
        data_loader = torch.utils.data.DataLoader(data_set, batch_size, shuffle=False, num_workers=num_workers)

        classes = self.determine_classes(data_loader)
        is_pinned = data_set.is_pinned()

        # Puts model in training mode.
        self.model.train()
        self.model = self.model.to(self.device)
        self.loss_function = self.loss_function.to(self.device)

        start = datetime.now()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(data_loader):
                # get the inputs; data is a list of [tensors, labels]
                tensors, labels = data
                tensors = tensors.to(self.device)

                # labels returned by the classifier are strings, we need to convert this to an int
                labels = torch.tensor([classes.index(label) for label in labels]).to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(tensors)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if self.verbosity_level >= 2 and i % 2000 == 1999:  # print every 2000 batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            if self.verbosity_level >= 1:
                epoch_time = datetime.now() - start
                eta = (datetime.now() + (epochs - epoch) * epoch_time).strftime('%H:%M:%S')
                print('Epoch {} finished in {}. ETA: {} -- Avg loss: {}'
                      .format(epoch + 1, epoch_time, eta, running_loss / len(data_loader.dataset)))
                start = datetime.now()
        if self.verbosity_level >= 1:
            print('Training finished, total time elapsed: {}'.format(datetime.now() - start))

    # save classifier object to .pkl file, can be retrieved with load_classifier()
    def save(self, file_path=None):
        if not isinstance(file_path, str):
            raise Exception("save method expects a file_path to be an instance of string")
        with open(file_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def set_verbosity_level(self, verbosity_level=0):
        if not isinstance(verbosity_level, int):
            raise Exception("verbosity_level parameter should be of an integer type")
        self.verbosity_level = verbosity_level

    def determine_classes(self, data_loader):
        if not isinstance(data_loader, torch.utils.data.DataLoader):
            raise Exception("data_loader parameter should be an instance of torch.utils.data.DataLoader")

        cnt = LabelCounter()
        for i, data in enumerate(data_loader):
            for label in data[1]:
                cnt.update(label)
        if self.verbosity_level:
            cnt.print()
        classes = list(cnt.dic.keys())
        # TODO this can be user provided so change it to reflect this
        self.model.fc = nn.Linear(self.model.fc.in_features, len(classes))
        return classes

    # Loads classifier object from .pkl file
    @staticmethod
    def from_pickle(pkl_file_path=None):
        if not isinstance(pkl_file_path, str):
            raise Exception("load_from_pickle method expects a pkl_file_path to be an instance of string")

        with open(pkl_file_path, 'rb') as input:
            cf = pickle.load(input)
            if not isinstance(cf, Classifier):
                raise Exception(pkl_file_path + ' does not contain a valid Classifier class object')
            return cf

    @staticmethod
    def validate_data_set_parameters(data_set=None, batch_size=4, epochs=20, num_workers=0):
        if not isinstance(epochs, int) or epochs < 0:
            raise Exception("epochs parameter should be a non-negative integer")

        if not isinstance(batch_size, int) or batch_size < 0:
            raise Exception("batch_size parameter should be a non-negative integer")

        if not isinstance(num_workers, int) or num_workers < 0:
            raise Exception("num_workers parameter should be a non-negative integer")

        if not isinstance(data_set, iterators.ABCIterator):
            raise Exception("data_set parameter should have a base class of data_iterators.ABCIterator")
