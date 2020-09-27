# We might want to generalize the Dicomo object part if we have time. It is nicer for the library that way.
# DataObject is such a class. Dicomo could extend DataObject
# DataObject can be anything, an Image with multiple labels, a signal with a signal label. etc. But it always has a
# tensor representing the data, and a dictionary of different labels and their values.

import gemicai as gem


class DataObject:
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels

