import pydicom

class LabelCounter:
    def __init__(self):
        self.dic = {}

    # I know this looks hideous but it prints a wonderfull table :)
    def __str__(self):
        s = 'label                          | frequency\n---------------------------------\n'
        t = 0
        for k, v in self.dic.items():
            t += v
            s += ('{:<30s} | {:>8d}\n'.format(k, v))
        return s + '\nTotal number of training images: {} \nTotal number of labels: {}'.format(t, len(self.dic.keys()))

    def update(self, s):
        if not isinstance(s, list) and not isinstance(s, str):
            raise Exception("LabelCounter update method expects a list or a string but " + str(type(s)) + " is given")

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
