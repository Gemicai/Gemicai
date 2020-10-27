# Everything concerning inspecting data
import gemicai as gem
import matplotlib.pyplot as plt
import pandas as pd
import torch


# TODO: Implement so that this function actually writes back to gemset
def correct_dataset(net: gem.Classifier, dataset: gem.GemicaiDataset):
    for tensor, label in dataset:
        prediction = net.classify(tensor)
        if prediction[0][0][0] != label:
            print('True label: {}\nPrediction : {}'.format(label, prediction))
            plt.imshow(torch.transpose(tensor, 0, 2))
            plt.show()
            input('Overwrite class?')
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


# Makes an excel file for the metadata and puts it in the specified dataset folder
def generate_metadata_excel(dataset_folder):
    generate_metadata_df(dataset_folder).to_excel('/home/nheinen/metadata.xlsx')


if __name__ == '__main__':
    generate_metadata_excel('/mnt/SharedStor/dataset/PT')


