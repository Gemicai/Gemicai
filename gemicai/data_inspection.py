# Everything concerning inspecting data
import gemicai as gem
import matplotlib.pyplot as plt
import pandas as pd
import torch


# TODO: Implement so that this function actually writes back to gemset
def correct_dataset(net: gem.Classifier, dataset: gem.GemicaiDataset):
    for index, tensor, label in enumerate(dataset):
        prediction = net.classify(tensor)
        if prediction[0][0][0] != label:
            print('True label: {}\nPrediction : {}'.format(label, prediction))
            plt.imshow(torch.transpose(tensor, 0, 2))
            plt.show()

            override = ''
            while override != 'Y' or override != 'N':
                override = input('Overwrite class? [Y/N]')

            if override == 'Y':
                dataset.modify(index, {dataset.labels[0]: prediction[0][0][0]})

            print('\n\n')


# Returns a pandas.DataFrame with meta data about a dicomos dataset
def generate_metadata_df(dataset_folder):
    # We want all dicomo labels
    dicomo_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
    ds = gem.DicomoDataset.get_dicomo_dataset(dataset_folder, dicomo_fields)
    df = []
    for data in ds:
        df.append(
            {
                'modality': data[1],
                'bpe': data[2],
                'studydes': data[3],
                'seriesdes': data[4],
            }
        )
    return pd.DataFrame(df)
