import gemicai.data_iterators as iterators
from gemicai.dicomo import LabelCounter
from torchsummary import summary
from datetime import datetime
import torch.nn as nn
import torchvision
import pickle
import torch


class Classifier:
    def __init__(self, base_model: nn.Module, enable_cuda=False, cuda_device=None):
        # Sets base model of the classifier
        self.model = base_model

        # select a correct cuda device
        if enable_cuda:
            if not torch.cuda.is_available():
                raise Exception("cuda is not available on this machine")

            device_name = "cuda"
            if cuda_device is not None:
                if not isinstance(cuda_device, int) or cuda_device < 0:
                    raise Exception("cuda_device parameter should be eiter set to None or be a non-negative number")
                device_name += ":" + cuda_device

            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cpu")

        # Default setting for training, can be overwritten in train() if save_as_default=True
        self.epochs = 20
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.batch_size = 4

        # Save location, will be stored after setting it once with save()
        self.file_path = None

        # Data loader will be set to None, has to bet set with set_data_loader() in order to train the classifier.
        self.data_loader = None

        # Data loader's metadata , used to calculate benchmarks.
        self.dl_train_directory = None
        # self.dl_total_images = None

        # The classes and classes counts will be initialized when setting a data loader.
        self.classes = None
        self.class_counts = None

        # Input shape of the tensors used by the classifier, only needed for keras like model summary
        self.input_shape = (3, 244, 244)

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
        testloader = get_data_loader(evaluation_directory, batch_size=self.batch_size)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                labels = torch.tensor([self.classes.index(label) for label in labels])
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Total: {} -- Correct: {} -- Accuracy: {}%'.format(total, correct, round(100 * correct / total, 2)))

    def set_data_loader(self, train_directory, verbosity=0, determine_classes=True):
        self.data_loader = get_data_loader(data_directory=train_directory, batch_size=self.batch_size)
        self.dl_train_directory = train_directory
        # This automatically determines all classes within the training data, and alters the classifer accordingly
        if determine_classes:
            cnt = LabelCounter()
            for i, data in self.data_loader:
                for label in data[1]:
                    cnt.update(label)
            if verbosity >= 1:
                cnt.print()
            self.class_counts = cnt
            self.classes = list(cnt.dic.keys())
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

    def train(self, epochs=None, loss_function=None, optimizer=None, verbosity=0, save_as_default=False):
        # Puts model in training mode.
        self.model.train()
        if epochs is None:
            epochs = self.epochs
        if loss_function is None:
            loss_function = self.loss_function
        if optimizer is None:
            optimizer = self.optimizer
        if save_as_default:
            self.epochs = epochs
            self.loss_function = loss_function
            self.optimizer = optimizer

        start = datetime.now()
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(self.data_loader):
                # get the inputs; data is a list of [tensors, labels]
                tensors, labels = data
                tensors = tensors.to(self.device)

                # labels returned by the classifier are strings, we need to convert this to an int
                labels = torch.tensor([self.classes.index(label) for label in labels])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(tensors)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

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
        # You can't store a generator object.
        self.data_loader = None
        with open(file_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


# Loads classifier objcet from .pkl file
def load_classifier(pkl_file_path, verbosity=0):
    with open(pkl_file_path, 'rb') as input:
        cf = pickle.load(input)
        assert isinstance(cf, Classifier), 'Not a valid Classifier'
        cf.set_data_loader(cf.dl_train_directory, determine_classes=False)
        if verbosity >= 1:
            print('Succesfully loaded classifier with classes: {}'.format(cf.classes))
        return cf


# os.path.join makes a platform dependent path (so both linux and windows works)
# =os.path.join('examples', 'compressed', 'CT', '000001.gz')
def get_data_loader(data_directory, use_pds=False, batch_size=4):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])

    if use_pds:
        # while creating PickleDataSet we pass a path to a pickle that hold the data
        # and a list of the fields that we want to extract from the dicomo object
        pickle_iter = iterators.PickledDicomoDataSet(data_directory, ['tensor', 'bpe'], transform)
    else:
        pickle_iter = iterators.PickledDicomoDataFolder(data_directory, ['tensor', 'bpe'], transform)

    # since we use a file with arbitrary number of dicomo objects we cannot parallelize loading data.
    # On the bright side we load only objects we currently need (batch_size) into memory
    return torch.utils.data.DataLoader(pickle_iter, batch_size, shuffle=False, num_workers=0)

