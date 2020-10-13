from abc import ABC, abstractmethod
from tabulate import tabulate
import openpyxl
import os


class OutputPolicy(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __del__(self):
        pass

    @abstractmethod
    def training_header(self):
        pass

    @abstractmethod
    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        pass

    @abstractmethod
    def training_finished(self, start, now):
        pass

    @abstractmethod
    def accuracy_summary_basic(self, total, correct, acc):
        pass

    @abstractmethod
    def accuracy_summary_extended(self, classes, class_total, class_correct):
        pass


class ToConsole(OutputPolicy):

    def __init__(self):
        None

    def __del__(self):
        None

    def training_header(self):
        print('| Epoch | Avg. loss | Train Acc. | Test Acc.  | Elapsed  |   ETA    |\n'
              '|-------+-----------+------------+------------+----------+----------|')

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        print('| {:5d} | {:.7f} | {:10s} | {:10s} | {:8s} | {} |'
              .format(epoch, running_loss / total, train_acc, test_acc, elapsed, eta))

    def training_finished(self, start, now):
        print('Training finished, total time elapsed: {}'.format(now - start))

    def accuracy_summary_basic(self, total, correct, acc):
        print('Total: {} -- Correct: {} -- Accuracy: {}%\n'.format(total, correct, acc))

    def accuracy_summary_extended(self, classes, class_total, class_correct):
        table = []
        for i, c in enumerate(classes):
            if class_total[i] != 0:
                class_acc = '{:.1f}%'.format(100 * class_correct[i] / class_total[i])
            else:
                class_acc = '-'
            table.append([c, class_total[i], class_correct[i], class_acc])
        print(tabulate(table, headers=['Class', 'Total', 'Correct', 'Acc'], tablefmt='orgtbl'), '\n')


class ToExcelFile(OutputPolicy):

    def __init__(self, file_path, override_content=False):
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
        cells = ["A", "B", "C", "D", "E", "F"]
        data_list = ["-----", "-----", "-----", "-----", "-----", "-----"]

        self._print_row(data_list, cells)
        self.row += 1

        self.workbook.save(self.file_path)

    def training_header(self):
        headers = ["Epoch", "Avg. loss", "Train Acc.", "Test Acc.", "Elapsed", "ETA"]
        cells = ["A", "B", "C", "D", "E", "F"]

        self._print_row(headers, cells)
        self.row += 1

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        cells = ["A", "B", "C", "D", "E", "F"]
        data_list = '{:5d} {:.7f} {:10s} {:10s} {:8s} {}'\
            .format(epoch, running_loss / total, train_acc, test_acc, elapsed, eta).split()

        self._print_row(data_list, cells)
        self.row += 1

    def training_finished(self, start, now):
        cells = ["A", "B"]
        data_list = ["Training finished in", "{}".format(now - start)]

        self._print_row(data_list, cells)
        self.row += 1

    def accuracy_summary_basic(self, total, correct, acc):
        cells = ["A", "B", "C"]
        header_list = ["Total", "Correct", "Accuracy"]
        data_list = '{} {} {}%'.format(total, correct, acc).split()

        self._print_row(header_list, cells)
        self.row += 1
        self._print_row(data_list, cells)
        self.row += 1

    def accuracy_summary_extended(self, classes, class_total, class_correct):
        cells = ["A", "B", "C", "D"]
        header_list = ["Class", "Total", "Correct", "Accuracy"]

        self._print_row(header_list, cells)
        self.row += 1

        for i, c in enumerate(classes):
            if class_total[i] != 0:
                class_acc = '{:.1f}%'.format(100 * class_correct[i] / class_total[i])
            else:
                class_acc = '-'
            data_list = [c, class_total[i], class_correct[i], class_acc]
            self._print_row(data_list, cells)
            self.row += 1

    def _print_row(self, data_list, cells):
        for index, data in enumerate(data_list):
            self.sheet[cells[index] + str(self.row)] = data


class ToConsoleAndExcelFile(ToConsole, ToExcelFile):

    def __init__(self, file_path, override_content=False):
        ToConsole.__init__(self)
        ToExcelFile.__init__(self, file_path, override_content)

    def __del__(self):
        ToExcelFile.__del__(self)

    def training_header(self):
        ToConsole.training_header(self)
        ToExcelFile.training_header(self)

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        ToConsole.training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta)
        ToExcelFile.training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta)

    def training_finished(self, start, now):
        ToConsole.training_finished(self, start, now)
        ToExcelFile.training_finished(self, start, now)

    def accuracy_summary_basic(self, total, correct, acc):
        ToConsole.accuracy_summary_basic(self, total, correct, acc)
        ToExcelFile.accuracy_summary_basic(self, total, correct, acc)

    def accuracy_summary_extended(self, classes, class_total, class_correct):
        ToConsole.accuracy_summary_extended(self, classes, class_total, class_correct)
        ToExcelFile.accuracy_summary_extended(self, classes, class_total, class_correct)
