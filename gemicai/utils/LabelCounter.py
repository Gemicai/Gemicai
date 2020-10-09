import pydicom
from tabulate import tabulate
from string import Template

class LabelCounter:
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
        if not isinstance(s, list) and not isinstance(s, str):
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
            if not isinstance(elem, pydicom.multival.MultiValue):
                check(elem)
                return
            for entry in elem:
                recurse(entry)

        if isinstance(s, str):
            check(s)
        else:
            for elem in s:
                recurse(elem)


# I'll drop this in gemicai.utils later
class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)
