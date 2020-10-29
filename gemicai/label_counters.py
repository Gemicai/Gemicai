"""This module contains label counters which are used by the data iterators in order to count distinct data
classes present in the dataset"""

from abc import ABC, abstractmethod
from tabulate import tabulate
import pydicom


class GemicaiLabelCounter(ABC):
    """Every custom label counter should extend this abstract base class"""

    @abstractmethod
    def __init__(self):
        """Sometimes there might be a need to do some special initialization"""
        pass

    @abstractmethod
    def __str__(self):
        """How class should behave when str() is called on it's instance"""
        pass

    @abstractmethod
    def update(self, labels):
        """This function is called whenever we have to count number of unique classes in a given input

        :param labels: in case of a user defined label counter it has to hold values to check against
        :type labels: any
        :return:
        """
        pass


class LabelCounter(GemicaiLabelCounter):
    """Gemicai's default label counter implementation"""

    def __init__(self, label=None):
        """Basic initialization"""
        self.label = label
        self.dic = {}

    def __str__(self):
        """Returns a string representation of a table which contains number of unique data classes and their names
        :return: string representation of a table
        """
        table = []
        for k, v in self.dic.items():
            table.append([k, v])
        if self.label is None:
            headers = ['Class', 'Frequency']
        else:
            headers = ['Class ({})'.format(self.label), 'Frequency']
        s = str(tabulate(table, headers=headers, tablefmt='orgtbl')) +\
            '\n\nTotal number of training images: {} \nTotal number of classes: {}\n'\
            .format(sum(self.dic.values()), len(self.dic.keys()))
        return s

    def update(self, labels):
        """This function checks if a given input is in it's internal mapping if not it is added to it and it's counter
        is set to one, otherwise if it is already present then the counter is incremented by one.

        :param labels: contains labels to count
        :type labels: Union[list, str, pydicom.valuerep.IS]
        """
        if labels is None:
            labels = 'None'
        if not isinstance(labels, list) and not isinstance(labels, str) and not isinstance(labels, pydicom.valuerep.IS):
            raise TypeError("LabelCounter update method expects a list or a string but " + str(type(labels)) +
                            " is given")

        # check whenever given label is already in our mapping
        def check(elem):
            if elem in self.dic.keys():
                self.dic[elem] += 1
            else:
                self.dic[elem] = 1
            return

        # recurse on a pydicom.multival.MultiValue or a list until we reach a value
        def recurse(elem):
            if isinstance(elem, str) or isinstance(elem, pydicom.valuerep.IS) or elem is None:
                check(str(elem))
            else:
                for entry in elem:
                    recurse(entry)

        if not isinstance(labels, list):
            check(str(labels))
        else:
            for elem in labels:
                recurse(elem)
