# Everything concerning inspecting data
import gemicai as gem
from matplotlib import pyplot as plt
import torch


# TODO: Implement so that this function actually writes back to gemset
def correct_dataset(net: gem.Classifier, dataset: gem.GemicaiDataset):
    for tensor, label in dataset:
        prediction = net.classify(tensor)
        if prediction[0][0] != label:
            print('True label: {}\nPrediction : {}'.format(label, prediction))
            plt.imshow(torch.transpose(tensor, 0, 2))
            plt.show()
            input('Overwrite class?')
            print('\n\n')



