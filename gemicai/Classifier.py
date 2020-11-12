"""This module contains a Classifier class which simplifies model training and evaluation process by abstracting away
many implementation details. As a result this module allows the user to save a lot of time by providing a default
implementation for many of the PyTorch options."""

import gemicai.classifier_functors as functr
import gemicai.data_iterators as iterators
import gemicai.output_policies as policy
from datetime import datetime
import torch.nn as nn
import pickle
import torch
from gemicai.utils import strfdelta
from operator import itemgetter
import gemicai as gem
from sklearn.metrics import confusion_matrix


class Classifier:
    """This class does all of the heavy lifting when it comes down to the model training, evaluation and tensor
    classification. During creation of this class it is possible to specify the following attributes:

    :param module: specifies a model to train, for more information about models themselves please refer
        to the https://pytorch.org/docs/stable/torchvision/models
    :type module: nn.Module
    :param classes: a list of classes present in the dataset, this will be used in order to modify model's last layer.
        For more information about how to obtain such a list please refer to the classes method of the
        gemicai.data_iterators.DicomoDataset
    :type classes: list
    :param layer_config: optional parameter containing a functor that can be used to modify a given model.
        For more information please refer to the gemicai.classifier_functors module
    :type layer_config: Optional[gemicai.classifier_functors.GEMICAIABCFunctor]
    :param loss_function: optional parameter containing a loss function used during a training.
    :type loss_function: Optional[nn.Module]
    :param optimizer: optional parameter containing an optimizer used during a training.
    :type optimizer: Optional[torch.optim.Optimizer]
    :param enable_cuda: if set to True the training will be done on the gpu otherwise the model will be
            trained on the cpu. Please note that training on a gpu is substantially faster.
    :type enable_cuda: Optional[bool]
    :param cuda_device: allows for selection of a particular cuda device if enable_cuda is set to True. PyTorch's
        Device ids start at 0 and are incremented by one.
    :type cuda_device: Optional[int]
    :raise RuntimeError: raised if the cuda device was selected but its not supported by the underlying machine
    :raises TypeError: raised if any of the parameters is of an invalid type
    """

    def __init__(self, module, classes, layer_config=None, loss_function=None, optimizer=None,
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
        elif not isinstance(loss_function, nn.Module):
             raise TypeError("Custom loss_function should have a base class of nn.Module")
        else:
            self.loss_function = loss_function

        # set a proper optimizer
        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.001, momentum=0.9)
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Custom optimizer should have a base class of torch.optim.Optimizer")
        else:
            self.optimizer = optimizer

    def train(self, dataset, batch_size=4, epochs=20, num_workers=0, pin_memory=False,
              verbosity=0, test_dataset=None, output_policy=policy.ToConsole()):
        """Used to train a model.

        :param dataset: dataset iterator used in order to train a model
        :type dataset: gemicai.data_iterators.GemicaiDataset
        :param batch_size: number (non-negative) of DataObject which will be feed into a classifier at once
        :type batch_size: int
        :param epochs: specifies how many training iterations (non-negative) to perform. One iteration goes over a
            whole dataset.
        :type epochs: int
        :param num_workers: number (non-negative) of worker threads used to load data from the dataset
        :type num_workers: int
        :param pin_memory: whenever memory pages should be pinned or not. If set to false there is a possibility
            that memory pages might be moved to a swap decreasing overall program's performance.
        :type pin_memory: bool
        :param verbosity: specifies verbosity (non-negative) of training/evaluation output. 0 - no output, 1 - basic
            output, 2 or more - extended output
        :type verbosity: int
        :param test_dataset: optional parameter, if a test_dataset iterator is passed and verbosity is set to at least 2
            it will be used in order to evaluate model's performance after a training epoch.
        :type test_dataset: Union[None, gemicai.data_iterators.GemicaiDataset]
        :param output_policy: specifies how and where to write the training statistics
        :type output_policy: gemicai.output_policies.OutputPolicy
        :raises TypeError: if passed arguments are not of a correct type or their values are outside of valid
            bounds this method will raise a TypeError exception.
        :raises ValueError: thrown whenever given data iterator object returns object containing more than two entries
        """
        Classifier.validate_dataset_parameters(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                                               pin_memory=pin_memory, test_dataset=test_dataset, verbosity=verbosity,
                                               output_policy=output_policy, epochs=epochs)

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
            output_policy.training_header()

        for epoch in range(epochs):
            running_loss, total = 0.0, 0
            for i, data in enumerate(data_loader):
                try:
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

                    running_loss += loss.item()
                    total += len(labels)
                except ValueError:
                    # This happens if provided label is not in classes.
                    pass
            if verbosity >= 1:
                epoch_time = datetime.now() - start
                start = datetime.now()
                eta = (datetime.now() + (epochs - epoch) * epoch_time).strftime('%H:%M:%S')
                train_acc, test_acc = '-', '-'
                # Evaluating models increases epoch time significantly
                if verbosity >= 2:
                    train_acc = str(self.evaluate(dataset)[0]) + '%'
                    if test_dataset is not None:
                        test_acc = str(self.evaluate(test_dataset)[0]) + '%'
                elapsed = strfdelta(epoch_time, '%H:%M:%S')
                output_policy.training_epoch_stats(epoch + 1, running_loss, total, train_acc, test_acc, elapsed, eta)
        if verbosity >= 1:
            output_policy.training_finished(start, datetime.now())

    def evaluate(self, dataset, batch_size=4, num_workers=0, pin_memory=False, verbosity=0,
                 output_policy=policy.ToConsole(), plot_cm=False):
        """Used to evaluate the model's performance on a provided dataset.

        :param dataset: dataset iterator used in order to evaluate a model's performance.
        :type dataset: gemicai.data_iterators.GemicaiDataset
        :param batch_size: number (non-negative) of DataObject which will be feed into a classifier at once
        :type batch_size: int
        :param num_workers: number (non-negative) of worker threads used to load data from the dataset
        :type num_workers: int
        :param pin_memory: whenever memory pages should be pinned or not. If set to false there is a possibility
            that memory pages might be moved to a swap decreasing overall program's performance.
        :type pin_memory: bool
        :param verbosity:  specifies verbosity (non-negative) of training/evaluation output. 0 - no output, 1 - basic
            output, 2 or more - extended output
        :type verbosity: int
        :param output_policy: specifies how and where to write the evaluation statistics
        :type output_policy: gemicai.output_policies.OutputPolicy
        :return: tuple of model's accuracy, number of total images and number of correctly classified images
        :raises TypeError: if passed arguments are not of a correct type or their values are outside of valid
            bounds this method will raise a TypeError exception.
        :raises ValueError: thrown whenever given data iterator object returns object containing more than two entries
        """
        Classifier.validate_dataset_parameters(dataset=dataset, batch_size=batch_size, num_workers=num_workers,
                                               epochs=0, pin_memory=pin_memory, test_dataset=None, verbosity=verbosity,
                                               output_policy=output_policy)

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
        true_labels, pred_labels = [], []
        with torch.no_grad():
            for data in data_loader:
                try:
                    images, labels = data
                    images = images.to(self.device)
                    labels = torch.tensor([self.classes.index(label) for label in labels]) \
                        .to(self.device, non_blocking=pin_memory)
                    outputs = self.module(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if plot_cm:
                        true_labels.extend(labels.tolist())
                        pred_labels.extend(predicted.tolist())
                    if verbosity >= 2:
                        for true_label, predicted_label in zip(labels, predicted):
                            if true_label == predicted_label:
                                class_correct[true_label] += 1
                            class_total[true_label] += 1
                except ValueError as e:
                    # This happens if provided label is not in classes.
                    pass
            if total == 0:
                acc = 'N/A'
            else:
                acc = round(100 * correct / total, 2)
            if verbosity == 1:
                output_policy.accuracy_summary_basic(total, correct, acc)
            if verbosity >= 2:
                output_policy.accuracy_summary_extended(self.classes, class_total, class_correct)
            if plot_cm:
                cm = confusion_matrix(true_labels, pred_labels)
                gem.utils.plot_confusion_matrix(cm, classes=self.classes, title='Confusion Matrix')
            return acc, total, correct

    def classify(self, tensor):
        """Takes in a tensor object and returns a list of predicted class types along with their certainties.
        :param tensor: tensor to classify
        :type tensor: torch.Tensor
        :return: list of predicted classes and their certainty
        :raise TypeError: raised if tensor does not have a torch.Tensor type
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor should be an instance of a torch.Tensor")
        self.module.to(self.device)
        tensor.to(self.device)
        if len(tensor.size()) == 3:
            tensor = torch.unsqueeze(tensor, 0).to(self.device)
        res, m = [], torch.nn.Softmax(dim=1)
        for classification in m(self.module(tensor).data):
            res.append(sorted(zip(self.classes, classification.tolist()), reverse=True, key=itemgetter(1))[:3])
        return res

    def save(self, file_path=None, zipped=False):
        """Saves current classifier object to the file system, it can be loaded back in using the
        gemicai.Classifier.Classifier.from_file method.

        :param file_path: a valid path to a file, it does not require a file to exist. Optionally .gemclas file
            extension can be appended to a file path like so /home/test/classifier.gemclas, if the extension is not
            present it will be added automatically.
        :type file_path: str
        :param zipped: whenever this object should be zipped or not
        :type zipped: bool
        :raises TypeError: file_path is not a str type
        """
        if not isinstance(file_path, str):
            raise TypeError("save method expects a file_path to be an instance of string")
        gem.io.save(file_path=file_path, obj=self, zipped=zipped)

    def set_device(self, enable_cuda=False, cuda_device=None):
        """Used in order to select a device on which model training will be done.

        :param enable_cuda: if set to True the training will be done on the gpu otherwise the model will be
            trained on the cpu. Please note that training on a gpu is substantially faster.
        :type enable_cuda: Optional[bool]
        :param cuda_device: allows for selection of a particular cuda device if enable_cuda is set to True. PyTorch's
            Device ids start at 0 and are incremented by one.
        :type cuda_device: Optional[int]
        :raise RuntimeError: raised if the cuda device was selected but its not supported by the underlying machine
        :raise TypeError: raised if any of the parameters has an invalid type
        """
        if not isinstance(enable_cuda, bool):
            raise TypeError("enable_cuda parameter should be a bool")

        self.enable_cuda = enable_cuda
        # select a correct cuda device
        if enable_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("cuda is not available on this machine")

            device_name = "cuda"
            if cuda_device is not None:
                if not isinstance(cuda_device, int) or cuda_device < 0:
                    raise TypeError("cuda_device parameter should be eiter set to None or be a non-negative integer")
                device_name += ":" + str(cuda_device)

            self.device = torch.device(device_name)
        else:
            self.device = torch.device("cpu")

    def set_trainable_layers(self, layers):
        """Sets specified layers to be either trainable or not.

        :param layers: list of tuples specifying which layers should be trainable or not, eg. [('fc', True), ...]. Where
            'fc' is a layer name and True specifies that it should be trainable. Note that instead of a layer
            name it is possible to pass 'all' in its place which will set every layer in the model to the
            specified mode, eg. [('all', False)] makes every layer untrainable.
        :type layers: list
        :raises TypeError: thrown if layers parameter has a wrong type
        """
        if not isinstance(layers, list):
            raise TypeError("set_trainable_layers method expects parameter layers to be a nonempty list "
                            "of tuples (layer_name: string, status: bool)")
        valid_layers = []

        # check whenever passed layers are valid/exist and set them if they are
        for name, param in self.module.named_parameters():
            name = '.'.join(name.split('.')[:-1])
            to_set = list(filter(lambda layer: layer[0] == name or layer[0] == "all", layers))
            if len(to_set):
                param.requires_grad = to_set[0][1]

    @staticmethod
    def from_file(file_path=None, zipped=False, enable_cuda=None):
        """Used to load a Classifier object from a given file

        :param file_path: a valid path to a file up to and including it's extension type.
        :type file_path: str
        :param zipped: whenever given file is zipped or not
        :type zipped: bool
        :param enable_cuda: Wheter or not cuda should be enabled,
        :type enable_cuda: bool
        :return: a valid Classifier object
        :raises TypeError: thrown if the given path is of an invalid format
        :raises Exception: thrown if Classifier object could not have been loaded in from the given file
        """
        if not isinstance(file_path, str):
            raise TypeError("load_from_pickle method expects a pkl_file_path to be an instance of string")
        net = gem.io.load(file_path, zipped=zipped)

        if enable_cuda is None:
            enable_cuda = torch.cuda.is_available()
        if net.enable_cuda != enable_cuda:
            net.set_device(enable_cuda=enable_cuda)

        return net

    @staticmethod
    def validate_dataset_parameters(dataset, batch_size, num_workers, pin_memory, test_dataset, verbosity,
                                    output_policy, epochs=1):
        """Called internally in order to validate passed arguments to the train and evaluate methods.

        :param dataset: dataset iterator used in order to train/evaluate a model
        :type dataset: gemicai.data_iterators.GemicaiDataset
        :param batch_size: number (non-negative) of DataObject which will be feed into a classifier at once
        :type batch_size: int
        :param num_workers: number (non-negative) of worker threads used to load data from the dataset
        :type num_workers: int
        :param pin_memory: whenever memory pages should be pinned or not. If set to false there is a possibility
            that memory pages might be moved to a swap decreasing overall program's performance.
        :type pin_memory: bool
        :param test_dataset: optional parameter, validates whenever a test_dataset iterator passed to the train
            function is a valid gemicai object
        :type test_dataset: Union[None, gemicai.data_iterators.GemicaiDataset]
        :param verbosity: specifies verbosity (non-negative) of training/evaluation output. 0 - no output, 1 - basic
            output, 2 or more - extended output
        :type verbosity: int
        :param output_policy: specifies how and where to write the training/evaluation output
        :type output_policy: gemicai.output_policies.OutputPolicy
        :param epochs: specifies how many training iterations (non-negative) to perform. One iteration goes over a
            whole dataset.
        :type epochs: int
        :raises TypeError: if passed arguments are not of a correct type or their values are outside of valid
            bounds this method will raise a TypeError exception.
        :raises ValueError: thrown whenever given data iterator object returns object containing more than two entries
        """
        if not isinstance(epochs, int) or epochs < 0:
            raise TypeError("epochs parameter should be a non-negative integer")

        if not isinstance(batch_size, int) or batch_size < 0:
            raise TypeError("batch_size parameter should be a non-negative integer")

        if not isinstance(num_workers, int) or num_workers < 0:
            raise TypeError("num_workers parameter should be a non-negative integer")

        if not isinstance(pin_memory, bool) or num_workers < 0:
            raise TypeError("pin_memory parameter should be a boolean")

        if not isinstance(dataset, iterators.GemicaiDataset):
            raise TypeError("dataset parameter should have a base class of data_iterators.GemicaiDataset")

        if len(next(iter(dataset))) != 2:
            raise ValueError('Specify what label should be classified. This dataset containts the labels {}. E.g. try'
                             ' again with dataset[\'{}\'] or dataset[0]'.format(dataset.labels, dataset.labels[0]))

        if not isinstance(test_dataset, iterators.GemicaiDataset) and test_dataset is not None:
            raise TypeError("test_dataset parameter should have a base class of data_iterators.GemicaiDataset "
                            "or be set to None")

        if not isinstance(verbosity, int) or verbosity < 0:
            raise TypeError("verbosity parameter should be a non-negative integer")

        if not isinstance(output_policy, policy.OutputPolicy):
            raise TypeError("output_policy parameter should have a base class of output_policies.OutputPolicy")
