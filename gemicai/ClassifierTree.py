from torchvision import models

import gemicai as gem
import torch
import copy
import time


class ClassifierTree:
    def __init__(self, root, base_dataset, classify, verbosity=0):
        if not isinstance(root, gem.Classifier):
            raise Exception('Root of ClassifierTree should be a gem.Classifier')
        if not isinstance(classify, list):
            raise Exception('classify should be a list of attributes the tree should classify')
        assert len(classify) >= 2, 'Classify should contain at least 2 classes.'

        self.root = root
        # self.constraints = constraints
        self.verbosity = verbosity
        self.classify = classify
        self.classifies = classify[0]

        # Children can either be a gemicai.Classifier or gemicai.ClassifierTree
        self.children = {}

        # Recursively sets up the ClassifierTree
        for c in self.root.classes:
            # Child inherits parent's dataset constraints, and gets extra contraint from its parent
            cons = {**self.root.dataset_constraints, self.classifies: c}
            child = gem.Classifier(module=copy.deepcopy(self.root.module),
                                   classes=base_dataset.subset(cons).classes(self.classify[1]),
                                   layer_config=copy.deepcopy(self.root.layer_config),
                                   loss_function=copy.deepcopy(self.root.loss_function),
                                   optimizer=copy.deepcopy(self.root.optimizer),
                                   verbosity_level=verbosity,
                                   enable_cuda=self.root.enable_cuda,
                                   # TODO cuda config when making a Tree
                                   # cuda_device= some device?
                                   dataset_constraints=cons)

            # If at the end of the tree, all  are leafs and therefore instances of gemicai.Classifier.
            # Else not the end of the tree, which means this child must be an instance of gemicai.ClassifierTree.
            if len(classify) == 2:
                self.children[c] = child
            else:
                self.children[c] = ClassifierTree(child, classify=classify[1:], constraints=cons, verbosity=verbosity)

    def train(self, data_set=None, batch_size=4, epochs=20, num_workers=0, pin_memory=False, redetermine_classes=False):
        self.root.train(dataset=data_set, batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                        pin_memory=pin_memory, redetermine_classes=redetermine_classes)
        for child in self.children:
            child.train(dataset=data_set, batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                        pin_memory=pin_memory, redetermine_classes=redetermine_classes)

    def __str__(self):
        s = 'Depth | Classifies | Classifiers | Avg. Classes\n'
        all_classifiers = self.get_all_classifiers()
        all_classifiers.append(self.root)
        tot_classifiers, tot_classes = 0, 0
        for i, cl in enumerate(self.classify):
            for c in all_classifiers:
                if c.dataset_config['object_fields'][1] == cl:
                    tot_classifiers += 1
                    tot_classes += len(c.classes)
            s += '{:<6d}|{:<12s}|{:>13d}|{:>14f}\n'.format(i, cl, tot_classifiers,
                                                           round(tot_classes / tot_classifiers, 1))
            tot_classifiers, tot_classes = 0, 0
        return s

    # Returns a set with all classifiers in the tree
    def get_all_classifiers(self):
        s = []
        for c in self.children.values():
            if isinstance(c, ClassifierTree):
                s += c.get_all_classifiers()
            else:
                s.append(c)
        return s


class ClassifierNode:
    pass

classify = ['bpe', 'studydes']
resnet18 = models.resnet18(pretrained=True)
ds = gem.PickledDicomoDataFolder(base_path='../examples/gzip/dx/train/', labels=['tensor', classify[0]])
net = gem.Classifier(resnet18, verbosity_level=1)
net.set_base_dataset(ds)
start = time.time()
tree = ClassifierTree(net, classify)
print('loading tree took {}'.format(time.time() - start))
print(tree)
print('printing tree took {}'.format(time.time() - start))
