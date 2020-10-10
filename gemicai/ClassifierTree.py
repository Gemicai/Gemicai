import gemicai as gem
import pickle
import copy
from tabulate import tabulate


class ClassifierTree:
    def __init__(self, labels, default_classifier=None, base_dataset=None, dataset_constraints={}, verbosity=0):
        if not isinstance(labels, list):
            raise Exception('classify should be a list of attributes the tree should classify')
        assert len(labels) >= 2, 'labels should contain at least 2 labels.'
        if not isinstance(base_dataset, gem.GemicaiDataset):
            raise Exception('base_dataset should be an instance of GemicaiDataset')
        if not isinstance(default_classifier, gem.Classifier):
            raise Exception('default classifier should be a gem.Classifier')

        # Set root of the tree
        classes = base_dataset.subset(dataset_constraints).classes(labels[0])
        self.root = ClassifierNode(default_classifier, labels[0], classes, dataset_constraints)

        # Children can either be a ClassifierNode or ClassifierTree
        self.labels = labels

        # Recursively sets up the ClassifierTree
        for c in classes:
            # Child inherits parent's dataset constraints, and gets extra constraint from its parent
            cons = {**self.root.dataset_constraints, labels[0]: c}

            # If at the end of the tree, all nodes are leafs and therefore instances of ClassifierNode.
            # Else not the end of the tree, which means this child must be an instance of ClassifierTree.
            if len(labels) == 2:
                child_classes = base_dataset.subset(cons).classes(labels[1])
                child = ClassifierNode(default_classifier, labels[1], child_classes)
            else:
                child = ClassifierTree(default_classifier, labels[1:], base_dataset, cons)

    def __str__(self):
        data, dic = [], {}
        for node in self.get_nodes():
            if node.label in dic.keys():
                cnt_classifiers, cnt_classes = dic[node.label]
                dic[node.label] = cnt_classifiers + 1, cnt_classes + len(node.classifier.classes)
            else:
                dic[node.label] = 1, len(node.classifier.classes)

        for label, (cnt_classifiers, cnt_classes) in dic.items():
            data.append([self.labels.index(label), label, cnt_classifiers,
                        '{:.1f}'.format(cnt_classes / cnt_classifiers)])
        return str(tabulate(data, headers=['Depth', 'Label', 'Classifiers', 'Avg. classes'], tablefmt='orgtbl'))

    def train(self, dataset, batch_size=4, epochs=20, num_workers=0, pin_memory=False, verbosity=0):
        self.root.train(dataset, epochs=epochs, num_workers=num_workers, pin_memory=pin_memory, verbosity=verbosity)
        for c in self.children:
            c.train(dataset, epochs=epochs, num_workers=num_workers, pin_memory=pin_memory, verbosity=verbosity)

    def evaluate(self):
        pass

    # Returns a list with all ClassifierNodes in tree
    def get_nodes(self):
        s = [self.root]
        for c in self.children.values():
            if isinstance(c, ClassifierTree):
                s += c.get_nodes()
            else:
                s.append(c)
        return s

    # Instantiates ClassifierTree object from ClassifierNode, or ClassifierNode filepath
    @staticmethod
    def from_node(node=None):
        if isinstance(node, str):
            with open(node, 'rb') as inp:
                node = pickle.load(inp)
        if not isinstance(node, ClassifierNode):
            raise TypeError('Not a valid ClassifierNode object')



# label: this is the label the node classifies e.g. 'BodyPartExamined'
# classes: this is a list of all possible classes a label can have. e.g. ['ABDOMEN','SKULL',etc.]
class ClassifierNode:
    def __init__(self, default_classifier, label, classes, dataset_constraints={}):
        self.classifier = gem.Classifier(module=copy.deepcopy(default_classifier.module), classes=classes,
                                         layer_config=copy.deepcopy(default_classifier.layer_config),
                                         loss_function=copy.deepcopy(default_classifier.loss_function),
                                         optimizer=copy.deepcopy(default_classifier.optimizer),
                                         enable_cuda=default_classifier.enable_cuda)
        # TODO cuda config for node and tree?
        self.label = label
        self.dataset_constraints = dataset_constraints

        # ClassifierNode will contain meta data about the classifier like how accurate it is. Otherwise training and
        # evaluating a big ClassifierTree is going to be impossible.
        self.accuracy = '77.77%'

    def train(self, dataset, epochs=20, num_workers=0, pin_memory=False, verbosity=0):
        dataset = dataset.subset(self.dataset_constraints)
        return self.classifier.train(dataset, epochs=epochs, num_workers=num_workers, pin_memory=pin_memory,
                                     verbosity=verbosity)

    def evaluate(self, dataset, num_workers=0, pin_memory=False, verbosity=0):
        dataset = dataset.subset(self.dataset_constraints)
        return self.classifier.evaluate(dataset, num_workers=num_workers, pin_memory=pin_memory, verbosity=verbosity)

    # save ClassifierNode object to .gemnode file, can be retrieved with from_file()
    def save(self, file_path):
        with open(file_path, 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    # Loads classifier object from .gemnode file
    @staticmethod
    def from_file(file_path):
        with open(file_path, 'rb') as inp:
            gemnode = pickle.load(inp)
            if not isinstance(gemnode, ClassifierNode):
                raise TypeError(file_path + ' does not contain a valid ClassifierNode object')
            return gemnode
