from abc import ABC, abstractmethod
from tabulate import tabulate


class OutputPolicy(ABC):

    @abstractmethod
    def __init__(self):
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
    def accuracy_summary_extended(self):
        pass


class OutputToConsole(OutputPolicy):

    def __init__(self):
        None

    def training_header(self):
        print('| Epoch | Avg. loss | Train Acc. | Test Acc.  | Elapsed  |   ETA    |\n'
              '|-------+-----------+------------+------------+----------+----------|')

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        print('| {:5d} | {:.7f} | {:10s} | {:10s} | {:8s} | {} |'
              .format(epoch + 1, running_loss / total, train_acc, test_acc, elapsed, eta))

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


class OutputToExcelFile(OutputPolicy):

    def __init__(self):
        None

    def training_header(self):
        None

    def training_epoch_stats(self, epoch, running_loss, total, train_acc, test_acc, elapsed, eta):
        None

    def training_finished(self, start, now):
        None

    def accuracy_summary_basic(self, total, correct, acc):
        None

    def accuracy_summary_extended(self):
        None