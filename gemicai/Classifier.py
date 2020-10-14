import gemicai.classifier_functors as functr
import gemicai.data_iterators as iterators
from datetime import datetime
import torch.nn as nn
import pickle
import torch
from gemicai.utils import strfdelta
from tabulate import tabulate
from operator import itemgetter

class Classifier:
    def __init__(self, module, classes=None, layer_config=None, loss_function=None, optimizer=None,
                 enable_cuda=False, cuda_device=None):
        # Sets base module of the classifier
        if not isinstance(module, nn.Module):
            raise TypeError("module_wrapper should extend a nn.Modules class")
        self.module = module
        self.device = None

        if not isinstance(classes, list):
            raise TypeError('Provide a valid list with classes')
        self.classes = classes

        # Sets a functor that allows us to configure the layers for training/evaluating
        if layer_config is None:
            self.layer_config = functr.DefaultLastLayerConfig()
        elif not isinstance(layer_config, functr.GEMICAIABCFunctor):
            raise TypeError("layer_config should extend a gemicai.classifier_functors.GEMICAIABCFunctor")
        else:
            self.layer_config = layer_config
        self.layer_config(self.module, self.classes)

        self.enable_cuda = enable_cuda
        self.set_device(enable_cuda, cuda_device)

        # set a proper loss function
        if loss_function is None:
            self.loss_function = nn.CrossEntropyLoss()
        # AttributeError: module 'torch.nn' has no attribute 'CrossEntropyLossImpl'
        # elif not isinstance(loss_function, nn.CrossEntropyLossImpl):
        #     raise Exception("Custom loss_function should have a base class of nn.CrossEntropyLossImpl")
        else:
            self.loss_function = loss_function

        # set a proper optimizer
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.001, momentum=0.9)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Custom optimizer should have a base class of torch.optim.Optimizer")
        else:
            self.optimizer = optimizer

    def train(self, dataset, batch_size=4, epochs=20, num_workers=0, pin_memory=False, verbosity=0, test_dataset=None):
        Classifier.validate_data_set_parameters(dataset, batch_size, epochs, num_workers, pin_memory)

        if not dataset.can_be_parallelized():
            num_workers = 0
        data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory)

        # Puts module in training mode.
        self.module.train()
        self.module = self.module.to(self.device, non_blocking=pin_memory)
        self.loss_function = self.loss_function.to(self.device, non_blocking=pin_memory)

        start = datetime.now()
        if verbosity >= 1:
            print('| Epoch | Avg. loss | Train Acc. | Test Acc.  | Elapsed  |   ETA    |\n'
                  '|-------+-----------+------------+------------+----------+----------|')
        for epoch in range(epochs):
            running_loss = 0.0
            total = 0
            for i, data in enumerate(data_loader):
                # get the inputs; data is a list of [tensors, labels]
                tensors, labels = data
                total += len(labels)

                tensors = tensors.to(self.device)
                # labels returned by the data loader are strings, we need to convert this to an int
                labels = torch.tensor([self.classes.index(label) for label in labels]) \
                    .to(self.device, non_blocking=pin_memory)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.module(tensors)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total += len(data[0])
                running_loss += loss.item()
            if verbosity >= 1:
                epoch_time = datetime.now() - start
                start = datetime.now()
                eta = (datetime.now() + (epochs - epoch) * epoch_time).strftime('%H:%M:%S')
                train_acc, test_acc = '-', '-'
                # Evaluating models increases epoch time significantly
                if verbosity >= 2:
                    train_acc = str(self.evaluate(dataset)) + '%'
                    if test_dataset is not None:
                        test_acc = str(self.evaluate(test_dataset)) + '%'
                elapsed = strfdelta(epoch_time, '%H:%M:%S')
                print('| {:5d} | {:.7f} | {:10s} | {:10s} | {:8s} | {} |'
                      .format(epoch + 1, running_loss / total, train_acc, test_acc, elapsed, eta))
        if verbosity >= 2:
            print('Training finished, total time elapsed: {}'.format(datetime.now() - start))

    def evaluate(self, dataset, batch_size=4, num_workers=0, pin_memory=False, verbosity=0):
        Classifier.validate_data_set_parameters(data_set=dataset, batch_size=batch_size,
                                                num_workers=num_workers, pin_memory=pin_memory)

        if not dataset.can_be_parallelized():
            num_workers = 0
        data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory)

        # puts module in evaluation mode.
        self.module.eval()
        self.module = self.module.to(self.device, non_blocking=pin_memory)

        correct, total = 0, 0
        class_correct = list(0. for _ in range(len(self.classes)))
        class_total = list(0. for _ in range(len(self.classes)))
        with torch.no_grad():
            for data in data_loader:
                images, labels = data
                images = images.to(self.device)
                labels = torch.tensor([self.classes.index(label) for label in labels]) \
                    .to(self.device, non_blocking=pin_memory)
                outputs = self.module(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if verbosity >= 2:
                    for true_label, predicted_label in zip(labels, predicted):
                        if true_label == predicted_label:
                            class_correct[true_label] += 1
                        class_total[true_label] += 1
            acc = round(100 * correct / total, 2)
            if verbosity >= 1:
                print('Total: {} -- Correct: {} -- Accuracy: {}%\n'.format(total, correct, acc))
            if verbosity >= 2:
                table = []
                for i, c in enumerate(self.classes):
                    if class_total[i] != 0:
                        class_acc = '{:.1f}%'.format(100 * class_correct[i] / class_total[i])
                    else:
                        class_acc = '-'
                    table.append([c, class_total[i], class_correct[i], class_acc])
                print(tabulate(table, headers=['Class', 'Total', 'Correct', 'Acc'], tablefmt='orgtbl'), '\n')
            return acc

    def classify(self, tensor):
        if len(tensor.size()) == 3:
            tensor = torch.unsqueeze(tensor, 0)
        res, m = [], torch.nn.Softmax(dim=1)
        for classification in m(self.module(tensor).data):
            res.append(sorted(zip(self.classes, classification.tolist()), reverse=True, key=itemgetter(1))[:3])
        return res

    # save classifier object to .pkl file, can be retrieved with load_classifier()
    def save(self, file_path=None):
        if not isinstance(file_path, str):
            raise TypeError("save method expects a file_path to be an instance of string")
        with open(file_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def set_device(self, enable_cuda=False, cuda_device=None):
        # select a correct cuda device
        if enable_cuda:
            if not torch.cuda.is_available():
                raise Exception("cuda is not available on this machine")

            device_name = "cuda"
            if cuda_device is not None:
                if not isinstance(cuda_device, int) or cuda_device < 0:
                    raise TypeError("cuda_device parameter should be eiter set to None or be a non-negative integer")
                device_name += ":" + str(cuda_device)

            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cpu")

    def set_trainable_layers(self, layers):
        if not isinstance(layers, list) and len(list) == 0:
            raise TypeError("set_trainable_layers method expects parameter layers to be a nonempty list "
                            "of tuples (layer_name: string, status: bool)")
        valid_layers = []

        # check whenever passed layers are valid/exist and set them if they are
        for name, param in self.module.named_parameters():
            name = '.'.join(name.split('.')[:-1])
            to_set = list(filter(lambda layer: layer[0] == name or layer[0] == "all", layers))
            if len(to_set):
                param.requires_grad = to_set[0][1]

    # Loads classifier object from .pkl file
    @staticmethod
    def from_file(pkl_file_path=None):
        if not isinstance(pkl_file_path, str):
            raise TypeError("load_from_pickle method expects a pkl_file_path to be an instance of string")

        with open(pkl_file_path, 'rb') as inp:
            cf = pickle.load(inp)
            if not isinstance(cf, Classifier):
                raise TypeError(pkl_file_path + ' does not contain a valid Classifier class object')
            return cf

    @staticmethod
    def validate_data_set_parameters(data_set=None, batch_size=4, epochs=20, num_workers=0, pin_memory=False):
        if not isinstance(epochs, int) or epochs < 0:
            raise TypeError("epochs parameter should be a non-negative integer")

        if not isinstance(batch_size, int) or batch_size < 0:
            raise TypeError("batch_size parameter should be a non-negative integer")

        if not isinstance(num_workers, int) or num_workers < 0:
            raise TypeError("num_workers parameter should be a non-negative integer")

        if not isinstance(pin_memory, bool) or num_workers < 0:
            raise TypeError("pin_memory parameter should be a boolean")

        if not isinstance(data_set, iterators.GemicaiDataset):
            raise TypeError("data_set parameter should have a base class of data_iterators.GemicaiDataset")
