from torchvision import models

import gemicai as gem
import torch
import copy


class ClassifierTree:
    def __init__(self, root, classify, data_directory, constraints={}, verbosity=0):
        if not isinstance(root, gem.Classifier):
            raise Exception('Root of ClassifierTree should be a gem.Classifier')
        if not isinstance(classify, list):
            raise Exception('classify should be a list of attributes the tree should classify')
        assert len(classify) >= 2, 'Classify should contain at least 2 classes.'
        if not isinstance(data_directory, str):
            raise Exception('data_direcotry should be a string')

        self.root = root
        self.constraints = constraints
        self.verbosity = verbosity
        self.classifies = classify[0]

        # The root of the tree should classify the first atrribute in classify, its children take care of the rest.
        self.root.determine_classes(gem.get_dicomo_data_loader(data_directory, ['tensor', self.classifies]))

        # Children can either be a gemicai.Classifier or gemicai.ClassifierTree
        # Sidenote: I think for now we don't need to keep track of parents but I might be wrong.
        self.children = {}

        # Recursively sets up the ClassifierTree
        for c in self.root.classes:
            cons = copy.deepcopy(self.constraints)
            cons[self.classifies] = c
            data_loader = gem.get_dicomo_data_loader(data_directory, ['tensor', classify[1]], contraints=cons)
            child = gem.Classifier(module=copy.deepcopy(self.root.module), layer_config=self.root.layer_config,
                                   loss_function=copy.deepcopy(self.root.loss_function), verbosity_level=verbosity,
                                   optimizer=copy.deepcopy(self.root.optimizer), enable_cuda=self.root.enable_cuda)
            child.determine_classes(data_loader)

            # If at the end of the tree, all  are leafs and therefore instances of gemicai.Classifier.
            # Else not the end of the tree, which means this child must be an instance of gemicai.ClassifierTree.
            if len(classify) == 2:
                self.root.children[c] = child
            else:
                self.root.children[c] = ClassifierTree(child, classify=classify[1:], data_directory=data_directory,
                                                       constraints=cons, verbosity=verbosity)

    # TODO: Should we give Classifier contraints?
    def train(self, data_set=None, batch_size=4, epochs=20, num_workers=0, pin_memory=False, redetermine_classes=False):
        self.root.train(data_set=data_set, batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                        pin_memory=pin_memory, redetermine_classes=redetermine_classes)
        for child in self.children:
            child.train(data_set=data_set, batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                        pin_memory=pin_memory, redetermine_classes=redetermine_classes)

    def __str__(self):
        depth = 0
        print('Depth | Classifies | Classifiers |  Avg. Classes')
        print()
        print('{} | ')
        return 'yeah'


resnet18 = models.resnet18(pretrained=True)
dl = gem.get_dicomo_data_loader('examples/gzip/dx/train/')
net = gem.Classifier(resnet18, dl, verbosity=1)
tree = ClassifierTree(net, 'studydes')
