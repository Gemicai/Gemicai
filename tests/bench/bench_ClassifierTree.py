from torchvision import models
import gemicai as gem
from datetime import datetime
import torch

tree_path = 'examples/tree'


def bench_initialize_tree():
    # Select the fields the tree should sequencially. The first item in the list will what the root of the tree
    # classifies. Initializing the Tree takes quite long, as it has to calculate the classes for every node.
    relevant_labels = ['Modality', 'BodyPartExamined', 'StudyDescription']
    resnet18 = models.resnet18(pretrained=True)
    ds = gem.DicomoDataset.get_dicomo_dataset('examples', labels=relevant_labels)
    net = gem.Classifier(resnet18, [], enable_cuda=False)
    start = datetime.now()
    tree = gem.ClassifierTree(default_classifier=net, labels=relevant_labels, base_dataset=ds, path=tree_path)
    print('Initializing tree took : {}'.format(gem.utils.strfdelta(datetime.now() - start, '%H:%M:%S')))
    print(tree)


def bench_train_tree():
    pass

print(models.mobilenet_v2(pretrained=True))
# tree = gem.ClassifierTree.from_dir(tree_path)
# print(tree)
# ds = gem.DicomoDataset.get_dicomo_dataset('examples', labels=['Modality', 'BodyPartExamined', 'StudyDescription'])
# ds_iter = iter(ds)
# tensors = torch.cat((torch.unsqueeze(next(ds_iter)[0], 0), torch.unsqueeze(next(ds_iter)[0], 0)))
# cls = tree.classify(tensors)
# print(cls)