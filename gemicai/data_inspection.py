# Everything concerning inspecting data
import gemicai as gem
import matplotlib as plt


def correct_dataset(net: gem.Classifier, dataset: gem.GemicaiDataset):
    for tensor, label in dataset:
        prediction = net.classify(tensor)
        if prediction[0][0] != label:
            print('True label: {}\nPrediction : {}'.format(label, prediction))
            plt.imshow(tensor)
            plt.show()
            input('Overwrite class?')



