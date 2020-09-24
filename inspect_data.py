import gemicai as gem
import pandas as pd


# Returns a pandas.DataFrame with meta data about a dicomos dataset
def generate_metadata_df(dataset_folder):
    # We want all dicomo labels
    dicomo_fields = ['modality', 'bpe', 'studydes', 'seriesdes', 'protocol']
    dl = gem.get_dicomo_data_loader(dataset_folder, dicomo_fields=dicomo_fields)
    df = []
    for data in dl:
        batch_size = len(data[0])
        for i in range(batch_size):
            df.append(
                {
                    'modality': data[0][i],
                    'bpe': data[1][i],
                    'studydes': data[2][i],
                    'seriesdes': data[3][i],
                    'protocol': data[4][i],
                }
            )
    return pd.DataFrame(df)


# Makes an excel file for the metadata and puts it in the specified dataset folder
def generate_metadata_excel(dataset_folder):
    generate_metadata_df(dataset_folder).to_excel(dataset_folder+'/metadata.xlsx')


generate_metadata_excel('examples/gzip/dx/test')
