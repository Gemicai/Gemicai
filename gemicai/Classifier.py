import gemicai.data_iterators as iterators
from gemicai.dicomo import LabelCounter
from torchsummary import summary
from datetime import datetime
import torch.nn as nn
import torchvision
import pickle
import torch


class Classifier:
    def __init__(self, base_model: nn.Module, data_loader: torch.data.DataLoader, enable_cuda=False, cuda_device=None,
                 loss_function=None, optimizer=None, classifies=None, verbosity=0):
        # Sets base model of the classifier
        self.model = base_model

        # You should set the data_loader with the funciton set_data_loader(), this automatically configures self.model
        # to work with the classes present in the data_loader. It also intializes self.classes and self.class_counts.
        self.data_loader = None
        self.classes = None
        self.class_counts = None

        self.set_data_loader(data_loader, verbosity=verbosity)

        # You can't pickle a generator object, and therefore self.data_loader cannot be pickled. These attributes store
        # data to recontruct the data_loader when
        self.dl_directory = None
        self.dl_batch_size = 4

        # select a correct cuda device
        self.enable_cuda = enable_cuda
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

        # Default setting for training
        self.epochs = 20

        # Default loss function and optimizer
        self.loss_function = loss_function if loss_function else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Save location, will be stored after setting it once with save().
        self.file_path = None

        # Input shape of the tensors used by the classifier, only needed for keras like model summary
        self.input_shape = (3, 244, 244)

        # Here you can store information about what the classifier clasifies. Used by gemicai.ClassifierTree
        self.classifies = classifies

    def set_trainable_layers(self, layers, boolean):
        for name, param in self.model.named_parameters():
            name = '.'.join(name.split('.')[:-1])
            if layers == 'all' or name in layers:
                param.requires_grad = boolean

    def summary(self):
        summary(self.model, self.input_shape)

    def evaluate(self, evaluation_directory, verbosity=0):
        # puts model in evaluation mode.
        self.model.eval()
        correct, total = 0, 0
        testloader = get_dicomo_data_loader(evaluation_directory, batch_size=self.dl_batch_size)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(self.device)
                labels = torch.tensor([self.classes.index(label) for label in labels]).to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Total: {} -- Correct: {} -- Accuracy: {}%'.format(total, correct, round(100 * correct / total, 2)))

    def set_data_loader(self, data_loader: torch.utils.data.DataLoader, verbosity=0, determine_classes=True):
        self.data_loader = data_loader
        self.dl_batch_size = self.data_loader.dl_batch_size

        self.dl_directory = self.data_loader.dataset.base_path
        if determine_classes:
            cnt = LabelCounter()
            for i, data in enumerate(self.data_loader):
                for label in data[1]:
                    cnt.update(label)
            if verbosity >= 1:
                cnt.print()
            self.class_counts = cnt
            self.classes = list(cnt.dic.keys())
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

    def train(self, epochs=None, verbosity=0):
        # Puts model in training mode.
        self.model.train()
        self.model.to(self.device)

        if epochs is None:
            epochs = self.epochs

        self.loss_function = self.loss_function.to(self.device)

        start = datetime.now()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.data_loader):
                # get the inputs; data is a list of [tensors, labels]
                tensors, labels = data
                tensors = tensors.to(self.device)

                # labels returned by the classifier are strings, we need to convert this to an int
                labels = torch.tensor([self.classes.index(label) for label in labels]).to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(tensors)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if verbosity >= 2 and i % 2000 == 1999:  # print every 2000 batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            if verbosity >= 1:
                epoch_time = datetime.now() - start
                eta = (datetime.now() + (self.epochs - epoch) * epoch_time).strftime('%H:%M:%S')
                print('Epoch {} finished in {}. ETA: {} -- Avg loss: {}'
                      .format(epoch + 1, epoch_time, eta, running_loss / len(self.data_loader.dataset)))
                start = datetime.now()
        if verbosity >= 1:
            print('Training finished, total time elapsed: {}'.format(datetime.now() - start))

    # save classifier object to .pkl file, can be retrieved with load_classifier()
    def save(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        else:
            self.file_path = file_path
        # You can't pickle a generator object.
        self.data_loader = None
        with open(file_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


# Loads classifier objcet from .pkl file
# FIXME: currently only works when the data_loader is a dicomo_data_loader
def load_classifier(pkl_file_path, verbosity=0):
    with open(pkl_file_path, 'rb') as inp:
        c = pickle.load(inp)
        assert isinstance(c, Classifier), 'Not a valid Classifier'
        dl = get_dicomo_data_loader(c.dl_directory, dicomo_fields=['tensor', c.classifies], batch_size=c.dl_batch_size)
        c.set_data_loader(dl, verbosity=verbosity, determine_classes=False)
        if verbosity >= 1:
            print('Succesfully loaded classifier with classes: {}'.format(c.classes))
        return c


# Returns a dicomo dataloader.
# FIXME: this function should probably not be in this file
def get_dicomo_data_loader(data_directory, dicomo_fields=['tensor', 'bpe'], batch_size=4):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
    pickle_iter = iterators.PickledDicomoDataFolder(data_directory, dicomo_fields, transform)
    # since we use a file with arbitrary number of dicomo objects we cannot parallelize loading data.
    # On the bright side we load only objects we currently need (batch_size) into memory
    return torch.utils.data.DataLoader(pickle_iter, batch_size, shuffle=False, num_workers=0)
