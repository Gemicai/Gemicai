from torchvision import models
import gemicai as gem
from datetime import datetime

tree_path = 'examples/tree'


def bench_initialize_tree():
    # Select the fields the tree should sequencially. The first item in the list will what the root of the tree
    # classifies. Initializing the Tree takes quite long, as it has to calculate the classes for every node.
    relevant_labels = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
    resnet18 = models.resnet18(pretrained=True)
    ds = gem.DicomoDataset.get_dicomo_dataset('examples', labels=relevant_labels)
    net = gem.Classifier(resnet18, [], enable_cuda=False)
    # ds.summarize('BodyPartExamined')
    start = datetime.now()
    tree = gem.ClassifierTree(default_classifier=net, labels=relevant_labels, base_dataset=ds, path=tree_path)
    print('Initializing tree took : {}'.format(gem.utils.strfdelta(datetime.now() - start, '%H:%M:%S')))
    print(tree)


def bench_train_tree():
    tree = gem.ClassifierTree.from_dir(tree_path)
    print(tree)


bench_train_tree()