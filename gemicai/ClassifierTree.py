from torchvision import models

import gemicai as gem
import torch


class ClassifierTree:
    def __init__(self, root, children_classify, verbosity=0):
        if not isinstance(root, gem.Classifier):
            raise Exception("Root of ClassifierTree should be a gem.Classifier")


        self.root = root
        self.verbosity = verbosity

        # Children can either be a gemicai.Classifier or gemicai.ClassifierTree
        # Sidenote: I think for now we don't need to keep track of parents but I might be wrong.
        self.children = {}

        # The attribute its children should classify
        self.children_classify = children_classify

        # Recursively sets up the ClassifierTree
        for c in self.root.classes:
            dl = gem.get_dicomo_data_loader(self.root.dl_directory, dicomo_fields=['tensor', children_classify[0]],
                                            batch_size=self.root.dl_batch_size)
            child = gem.Classifier(self.root.model, dl, enable_cuda=self.root.enable_cuda,
                                   loss_function=self.root.loss_function, optimizer=self.root.optimizer,
                                   classifies=children_classify[0], verbosity=verbosity)

            # If at the end of the tree, all  are leafs and therefore instances of gemicai.Classifier.
            # Else not the end of the tree, which means this child must be an instance of gemicai.ClassifierTree.
            if len(children_classify) == 1:
                self.root.children[c] = child
            else:
                self.root.children[c] = ClassifierTree(child, children_classify[1:], verbosity=verbosity)

    # TODO write function that prints information about the tree
    def summary(self):
        pass


resnet18 = models.resnet18(pretrained=True)
dl = gem.get_dicomo_data_loader('examples/gzip/dx/train/')
net = gem.Classifier(resnet18, dl, verbosity=1)
tree = ClassifierTree(net, 'studydes')
