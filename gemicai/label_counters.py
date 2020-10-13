from abc import ABC, abstractmethod
from tabulate import tabulate
import pydicom


class GemicaiLabelCounter(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def update(self, s):
        pass


class LabelCounter(GemicaiLabelCounter):
    def __init__(self):
        self.dic = {}

    def __str__(self):
        table = []
        for k, v in self.dic.items():
            table.append([k, v])
        s = str(tabulate(table, headers=['Class', 'Frequency'], tablefmt='orgtbl')) +\
            '\nTotal number of training images: {} \nTotal number of classes: {}\n'\
            .format(sum(self.dic.values()), len(self.dic.keys()))
        return s

    def update(self, s):
        if not isinstance(s, list) and not isinstance(s, str) and not isinstance(s, pydicom.valuerep.IS):
            raise TypeError("LabelCounter update method expects a list or a string but " + str(type(s)) + " is given")

        # check whenever given label is already in our mapping
        def check(elem):
            if elem in self.dic.keys():
                self.dic[elem] += 1
            else:
                self.dic[elem] = 1
            return

        # recurse on a pydicom.multival.MultiValue or a list until we reach a value
        def recurse(elem):
            if isinstance(elem, str) or isinstance(elem, pydicom.valuerep.IS):
                check(str(elem))
            else:
                for entry in elem:
                    recurse(entry)

        if not isinstance(s, list):
            check(str(s))
        else:
            for elem in s:
                recurse(elem)
