import gemicai as gem
import os


# Bit of a hack but works for now I guess
def get_demo_gemset(labels):
    path = os.path.abspath(__file__)
    path = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    path = os.path.join(path, 'examples/gemset/demo/')
    return gem.DicomoDataset.get_dicomo_dataset(path, labels)