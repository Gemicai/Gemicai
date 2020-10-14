import gemicai as gem
import pickle
import copy
from tabulate import tabulate
import os


class ClassifierTree:
    def __init__(self, default_classifier, labels, base_dataset, path, dataset_constraints={}, root=None):
        # Retrieving ClassifierTree from a root node
        if labels is None and default_classifier is None and base_dataset is None and path is None:
            if not isinstance(root, ClassifierNode):
                raise Exception('root should be of type ClassifierNode')
            self.root = root
            self.path = os.path.dirname(self.root.file_path)
        # Initializing ClassifierTree
        else:
            if not isinstance(labels, list):
                raise Exception('classify should be a list of attributes the tree should classify')
            assert len(labels) >= 2, 'labels should contain at least 2 labels.'
            if not isinstance(base_dataset, gem.GemicaiDataset):
                raise Exception('base_dataset should be an instance of GemicaiDataset')
            if not isinstance(default_classifier, gem.Classifier):
                raise Exception('default classifier should be a gem.Classifier')
            if not os.path.isdir(path):
                raise Exception('path should be a valid directory')
            self.path = path

            # Set root of the tree
            classes = base_dataset.subset(dataset_constraints).classes(labels[0])
            root_file_path = os.path.join(self.path, labels[0])
            self.root = ClassifierNode(default_classifier, labels[0], classes, root_file_path, dataset_constraints)
            self.root.save()

            # Recursively sets up the ClassifierTree
            for c in classes:
                # Child inherits parent's dataset constraints, and gets extra constraint from its parent
                cons = {**self.root.dataset_constraints, labels[0]: c}

                # If at the end of the tree, all nodes are leafs and therefore instances of ClassifierNode.
                # Else not the end of the tree, which means this child must be an instance of ClassifierTree.
                child_dir = os.path.join(self.path, c)
                if not os.path.exists(child_dir):
                    os.makedirs(child_dir)

                if len(labels) == 2:
                    child_classes = base_dataset.subset(cons).classes(labels[1])
                    child_path = os.path.join(child_dir, labels[1])
                    ClassifierNode(default_classifier, labels[1], child_classes, child_path, cons).save()
                else:
                    ClassifierTree(default_classifier, labels[1:], base_dataset, child_dir, cons)

    def __str__(self):
        data, dic = [], {}
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('.gemnode'):
                    node = ClassifierNode.from_file(os.path.join(root, file))
                    if node.label in dic.keys():
                        cnt_classifiers, cnt_classes = dic[node.label]
                        dic[node.label] = cnt_classifiers + 1, cnt_classes + len(node.classifier.classes)
                    else:
                        dic[node.label] = 1, len(node.classifier.classes)

        labels = list(dic.keys())
        for label, (cnt_classifiers, cnt_classes) in dic.items():
            data.append([labels.index(label), label, cnt_classifiers, '{:.1f}'.format(cnt_classes / cnt_classifiers)])
        return str(tabulate(data, headers=['Depth', 'Label', 'Classifiers', 'Avg. classes'], tablefmt='orgtbl'))

    def train(self, dataset, epochs=20, num_workers=0, pin_memory=False, verbosity=0):
        self.root.train(dataset, epochs=epochs, num_workers=num_workers, pin_memory=pin_memory, verbosity=verbosity)
        for child in self.root.children():
            node = ClassifierNode.from_file(child)
            node.train(dataset, epochs=epochs, num_workers=num_workers, pin_memory=pin_memory, verbosity=verbosity)

    def evaluate(self):
        # TODO
        pass

    def classify(self, tensor):
        predicted = self.root.classify(tensor)
        if self.root.is_leaf():
            return {self.root.label: predicted}
        else:
            child = gem.ClassifierTree.from_dir(os.path.join(self.path, predicted[0][0]))
            return {self.root.label: predicted, **child.classify(tensor)}

    # Instantiates ClassifierTree object from ClassifierNode, or ClassifierNode filepath
    @staticmethod
    def from_node(node=None):
        if isinstance(node, str):
            with open(node, 'rb') as inp:
                node = pickle.load(inp)
        if not isinstance(node, ClassifierNode):
            raise TypeError('Not a valid ClassifierNode object')
        return ClassifierTree(None, None, None, None, root=node)

    @staticmethod
    def from_dir(directory):
        for f in os.listdir(directory):
            if f.endswith('.gemnode'):
                return ClassifierTree.from_node(node=os.path.join(directory, f))


# label: this is the label the node classifies e.g. 'BodyPartExamined'
# classes: this is a list of all possible classes a label can have. e.g. ['ABDOMEN','SKULL',etc.]
class ClassifierNode:
    def __init__(self, default_classifier, label, classes, file_path, dataset_constraints={}):
        self.classifier = gem.Classifier(module=copy.deepcopy(default_classifier.module), classes=classes,
                                         layer_config=copy.deepcopy(default_classifier.layer_config),
                                         loss_function=copy.deepcopy(default_classifier.loss_function),
                                         optimizer=copy.deepcopy(default_classifier.optimizer),
                                         enable_cuda=default_classifier.enable_cuda)
        self.label = label
        self.file_path = file_path
        self.dataset_constraints = dataset_constraints

        # ClassifierNode will contain meta data about the classifier like how accurate it is. Otherwise training and
        # evaluating a big ClassifierTree is going to be impossible.
        self.accuracy = '77.77%'

    def train(self, dataset, epochs=20, num_workers=0, pin_memory=False, verbosity=0):
        dataset = dataset.subset(self.dataset_constraints)
        self.classifier.train(dataset, epochs=epochs, num_workers=num_workers, pin_memory=pin_memory,
                              verbosity=verbosity)

    def evaluate(self, dataset, num_workers=0, pin_memory=False, verbosity=0):
        dataset = dataset.subset(self.dataset_constraints)
        self.accuracy = self.classifier.evaluate(dataset, num_workers=num_workers, pin_memory=pin_memory,
                                                 verbosity=verbosity)

    # ClassifierNode can only classify 1 tensor at a time.
    def classify(self, tensor):
        return self.classifier.classify(tensor)[0]

    # Return list off all immediate children
    def children(self):
        children, dirname = [], os.path.dirname(self.file_path)
        for d in os.listdir(dirname):
            if os.path.isdir(os.path.join(dirname, d)):
                children.append(os.path.join(dirname, d))
        return children

    # If the node has no children its a leaf node.
    def is_leaf(self):
        return len(self.children()) == 0

    # save ClassifierNode object to .gemnode file, can be retrieved with from_file()
    def save(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        else:
            self.file_path = file_path
        with open(file_path+'.gemnode', 'wb') as out:
            pickle.dump(self, out, pickle.HIGHEST_PROTOCOL)

    # Loads classifier object from .gemnode file
    @staticmethod
    def from_file(file_path):
        with open(file_path, 'rb') as inp:
            gemnode = pickle.load(inp)
            if not isinstance(gemnode, ClassifierNode):
                raise TypeError(file_path + ' does not contain a valid ClassifierNode object')
            return gemnode
