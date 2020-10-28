"""This module contains output policies. Such policy can be supplied as an optional parameter during model
training or evaluation in order to log, save, or print statistics related to the model's performance.
"""

from abc import ABC, abstractmethod
from tabulate import tabulate
import openpyxl
import os


class OutputPolicy(ABC):
    """Every custom policy should extend this abstract base class."""

    @abstractmethod
    def __init__(self):
        """Executed when the class is created, can be useful if some special instantiation is needed."""
        pass

    @abstractmethod
    def __del__(self):
        """Executed when the class goes out of scope, can be useful if some special cleanup is needed."""
        pass

    @abstractmethod
    def training_header(self):
        """Class implementing this method should make sure to specify how to handle a training header call.
        This call happens once before training. It's purpose is to beautify training_epoch_stats output by providing
        some context into what categories of data the user is looking at.
        """
        pass

    @abstractmethod
    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        """This function is called every time an epoch ends, all of the important training statistics are taken as
        an input.

        :param epoch: current training epoch epoch
        :type epoch: int
        :param running_loss: current total running loss of the model
        :type running_loss: float
        :param total: number of the images that model has trained on
        :type total: int
        :param train_acc: models accuracy on the provided train dataset
        :type train_acc: str
        :param test_acc: models accuracy on the optionally provided eval dataset
        :type test_acc: str
        :param elapsed: total time it took to run the epoch
        :type elapsed: str
        :param eta: an estimated time when training will end
        :type eta: str
        """
        pass

    @abstractmethod
    def training_finished(self, start, now):
        """Called once when training has finished.

        :param start: time when the training has started
        :type start: datetime.datetime
        :param now: current time
        :type now: datetime.datetime
        """
        pass

    @abstractmethod
    def accuracy_summary_basic(self, total, correct, acc):
        """Called after a model evaluation finishes if verbosity is set to 1.

        :param total: number of total objects model was evaluated on
        :type total: int
        :param correct: number of correctly classified objects
        :type correct: int
        :param acc: overall accuracy of the model
        :type acc: float
        """
        pass

    @abstractmethod
    def accuracy_summary_extended(self, classes, class_total, class_correct):
        """Called after a model evaluation finishes if verbosity is equal or greater than 2.

        :param classes: list with class names on which model was evaluated
        :type classes: list
        :param class_total: list with a number of classes on which model was evaluated
        :type class_total: list
        :param class_correct: list with a number of properly classified classes
        :type class_correct: list
        """
        pass


class ToConsole(OutputPolicy):
    """This policy allows to output training statistics to the console"""

    def __init__(self):
        None

    def __del__(self):
        None

    def training_header(self):
        """Prints a training header to the console"""
        print('| Epoch | Avg. loss | Train Acc. | Test Acc.  | Elapsed  |   ETA    |\n'
              '|-------+-----------+------------+------------+----------+----------|')

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        """This function is called every time an epoch ends, all of the important training statistics are taken as
        an input and outputted to the console.

       :param epoch: current training epoch epoch
       :type epoch: int
       :param running_loss: current total running loss of the model
       :type running_loss: float
       :param total: number of the images that model has trained on
       :type total: int
       :param train_acc: models accuracy on the provided train dataset
       :type train_acc: str
       :param test_acc: models accuracy on the optionally provided eval dataset
       :type test_acc: str
       :param elapsed: total time it took to run the epoch
       :type elapsed: str
       :param eta: an estimated time when training will end
       :type eta: str
       """
        print('| {:5d} | {:.7f} | {:10s} | {:10s} | {:8s} | {} |'
              .format(epoch, running_loss / total, train_acc, test_acc, elapsed, eta))

    def training_finished(self, start, now):
        """Outputs elapsed training time to the console.

        :param start: time when the training has started
        :type start: datetime.datetime
        :param now: current time
        :type now: datetime.datetime
        """
        print('Training finished, total time elapsed: {}'.format(now - start))

    def accuracy_summary_basic(self, total, correct, acc):
        """Outputs model evaluation statistics to the console if verbosity is set to 1.

        :param total: number of total objects model was evaluated on
        :type total: int
        :param correct: number of correctly classified objects
        :type correct: int
        :param acc: overall accuracy of the model
        :type acc: float
        """
        print('Total: {} -- Correct: {} -- Accuracy: {}%\n'.format(total, correct, acc))

    def accuracy_summary_extended(self, classes, class_total, class_correct):
        """Outputs model evaluation statistics to the console if  verbosity is equal or greater than 2.

        :param classes: list with class names on which model was evaluated
        :type classes: list
        :param class_total: list with a number of classes on which model was evaluated
        :type class_total: list
        :param class_correct: list with a number of properly classified classes
        :type class_correct: list
        """
        print('| {} | {} | {} |'.format(type(classes), type(class_total), type(class_correct)))
        table = []
        for i, c in enumerate(classes):
            if class_total[i] != 0:
                class_acc = '{:.1f}%'.format(100 * class_correct[i] / class_total[i])
            else:
                class_acc = '-'
            table.append([c, class_total[i], class_correct[i], class_acc])
        print(tabulate(table, headers=['Class', 'Total', 'Correct', 'Acc'], tablefmt='orgtbl'), '\n')


class ToExcelFile(OutputPolicy):
    """This policy allows to output training statistics to the excel file"""

    def __init__(self, file_path, override_content=False):
        """Constructs ToExcelFile and opens/creates an excel file.

        :param file_path: path to an excel file
        :type file_path: str
        :param override_content: if file is present should we override its content
        :type override_content: bool
        """
        if not isinstance(file_path, str):
            raise TypeError("file_path parameter should be a string")
        if not isinstance(override_content, bool):
            raise TypeError("override_content parameter should be a bool")

        self.file_path = file_path
        self.row = 1

        if os.path.isfile(self.file_path):
            self.workbook = openpyxl.load_workbook(self.file_path)
        else:
            self.workbook = openpyxl.Workbook()
            self.workbook.save(self.file_path)
        self.sheet = self.workbook.active

        # check whenever A1 has been written to if yes go down until we find a Ax that is free
        if not override_content:
            while self.sheet["A" + str(self.row)].value is not None:
                self.row += 1

    def __del__(self):
        """Outputs a delimiter line to the excel file and saves it"""
        cells = ["A", "B", "C", "D", "E", "F"]
        data_list = ["-----", "-----", "-----", "-----", "-----", "-----"]
        self.print_row(data_list, cells)
        self.workbook.save(self.file_path)

    def training_header(self):
        """Outputs a training header to the excel file"""
        headers = ["Epoch", "Avg. loss", "Train Acc.", "Test Acc.", "Elapsed", "ETA"]
        cells = ["A", "B", "C", "D", "E", "F"]

        self.print_row(headers, cells)

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        """This function is called every time an epoch ends, all of the important training statistics are taken as
               an input and outputted to the specified excel file.

       :param epoch: current training epoch epoch
       :type epoch: int
       :param running_loss: current total running loss of the model
       :type running_loss: float
       :param total: number of the images that model has trained on
       :type total: int
       :param train_acc: models accuracy on the provided train dataset
       :type train_acc: str
       :param test_acc: models accuracy on the optionally provided eval dataset
       :type test_acc: str
       :param elapsed: total time it took to run the epoch
       :type elapsed: str
       :param eta: an estimated time when training will end
       :type eta: str
       """
        cells = ["A", "B", "C", "D", "E", "F"]
        data_list = '{:5d} {:.7f} {:10s} {:10s} {:8s} {}'\
            .format(epoch, running_loss / total, train_acc, test_acc, elapsed, eta).split()
        self.print_row(data_list, cells)

    def training_finished(self, start, now):
        """Called once when training has finished, it outputs the elapsed time to the specified excel file.

        :param start: time when the training has started
        :type start: datetime.datetime
        :param now: current time
        :type now: datetime.datetime
        """
        cells = ["A", "B"]
        data_list = ["Training finished in", "{}".format(now - start)]
        self.print_row(data_list, cells)

    def accuracy_summary_basic(self, total, correct, acc):
        """Called after a model evaluation finishes if verbosity is set to 1. Outputs a training statistics to the
        specified excel file.

        :param total: number of total objects model was evaluated on
        :type total: int
        :param correct: number of correctly classified objects
        :type correct: int
        :param acc: overall accuracy of the model
        :type acc: float
        """
        cells = ["A", "B", "C"]
        header_list = ["Total", "Correct", "Accuracy"]
        data_list = '{} {} {}%'.format(total, correct, acc).split()
        self.print_row(header_list, cells)
        self.print_row(data_list, cells)

    def accuracy_summary_extended(self, classes, class_total, class_correct):
        """Called after a model evaluation finishes if verbosity is equal or greater than 2.
        Outputs a training statistics to the specified excel file.

        :param classes: list with class names on which model was evaluated
        :type classes: list
        :param class_total: list with a number of classes on which model was evaluated
        :type class_total: list
        :param class_correct: list with a number of properly classified classes
        :type class_correct: list
        """

        cells = ["A", "B", "C", "D"]
        header_list = ["Class", "Total", "Correct", "Accuracy"]
        self.print_row(header_list, cells)

        for i, c in enumerate(classes):
            if class_total[i] != 0:
                class_acc = '{:.1f}%'.format(100 * class_correct[i] / class_total[i])
            else:
                class_acc = '-'
            data_list = [c, class_total[i], class_correct[i], class_acc]
            self.print_row(data_list, cells)

    def print_row(self, data_list, cells):
        """This function writes each entry from data_list into a separate cell.

        :param data_list: list with data to be written
        :type data_list: list
        :param cells: list with column names. Should have at least as many entries as data_list.
        :type cells: list
        :return:
        """
        for index, data in enumerate(data_list):
            self.sheet[cells[index] + str(self.row)] = data
        self.workbook.save(self.file_path)
        self.row += 1


class ToConsoleAndExcelFile(ToConsole, ToExcelFile):
    """This output policy is a composition of ToConsole and ToExcelFile policies."""

    def __init__(self, file_path, override_content=False):
        """Constructs ToConsoleAndExcelFile and opens/creates a specified excel file.

        :param file_path: path to an excel file
        :type file_path: str
        :param override_content: if file is present should we override its content
        :type override_content: bool
        """
        ToConsole.__init__(self)
        ToExcelFile.__init__(self, file_path, override_content)

    def __del__(self):
        """Outputs a delimiter line to the excel file and saves it"""
        ToExcelFile.__del__(self)

    def training_header(self):
        """Outputs training header to the console and the specified excel file"""
        ToConsole.training_header(self)
        ToExcelFile.training_header(self)

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        """This function is called every time an epoch ends, all of the important training statistics are taken
        as an input and outputs them to the console and the specified excel file.

       :param epoch: current training epoch epoch
       :type epoch: int
       :param running_loss: current total running loss of the model
       :type running_loss: float
       :param total: number of the images that model has trained on
       :type total: int
       :param train_acc: models accuracy on the provided train dataset
       :type train_acc: str
       :param test_acc: models accuracy on the optionally provided eval dataset
       :type test_acc: str
       :param elapsed: total time it took to run the epoch
       :type elapsed: str
       :param eta: an estimated time when training will end
       :type eta: str
       """
        ToConsole.training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta)
        ToExcelFile.training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta)

    def training_finished(self, start, now):
        """Called once when training has finished. Outputs elapsed time to the console and the specified excel file.

        :param start: time when the training has started
        :type start: datetime.datetime
        :param now: current time
        :type now: datetime.datetime
        """
        ToConsole.training_finished(self, start, now)
        ToExcelFile.training_finished(self, start, now)

    def accuracy_summary_basic(self, total, correct, acc):
        """Called after a model evaluation finishes if verbosity is set to 1. Outputs the training statistics to
        the console and to the specified excel file.

        :param total: number of total objects model was evaluated on
        :type total: int
        :param correct: number of correctly classified objects
        :type correct: int
        :param acc: overall accuracy of the model
        :type acc: float
        """
        ToConsole.accuracy_summary_basic(self, total, correct, acc)
        ToExcelFile.accuracy_summary_basic(self, total, correct, acc)

    def accuracy_summary_extended(self, classes, class_total, class_correct):
        """Called after a model evaluation finishes if verbosity is equal or greater than 2. Outputs the training
        statistics to the console and to the specified excel file.

        :param classes: list with class names on which model was evaluated
        :type classes: list
        :param class_total: list with a number of classes on which model was evaluated
        :type class_total: list
        :param class_correct: list with a number of properly classified classes
        :type class_correct: list
        """
        ToConsole.accuracy_summary_extended(self, classes, class_total, class_correct)
        ToExcelFile.accuracy_summary_extended(self, classes, class_total, class_correct)
