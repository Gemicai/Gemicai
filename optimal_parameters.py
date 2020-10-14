import torchvision.models as models
import gemicai as gem
import torch
import os


dataset_all = os.path.join("examples", "gzip", "CT")
dataset_pick_middle = os.path.join("examples", "gzip", "CT")
dataset_common_modalities = os.path.join("examples", "gzip", "CT")

dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
current_field = dicom_fields[0]
excel_file_name = "test.xlsx"

train_set = gem.DicomoDataset.get_dicomo_dataset(dataset_common_modalities, labels=[current_field])
eval_set = gem.DicomoDataset.get_dicomo_dataset(dataset_pick_middle, labels=[current_field])
resnet18 = models.resnet18(pretrained=True)


def summarize_sets():
    train_set.summarize(current_field)
    eval_set.summarize(current_field)


def log_train_options(data_list):
    excel = gem.ToExcelFile(excel_file_name)
    excel.print_row(data_list, ["A", "B", "C", "D", "E", "F"])


def train(field_list, value_list, modality):
    net = gem.Classifier(resnet18, train_set.classes(modality), enable_cuda=True)

    # set correct net params
    value_types = []
    if len(field_list):

        for index, field in enumerate(field_list):
            # check if the field exists if not just throw and terminate, a small anti oopsie feature
            getattr(net, field)
            setattr(net, field, value_list[index])
            value_types += [str(value_list[index])]

    # log train parameters to an excel file
    if len(value_types):
        log_train_options(value_types)
    else:
        log_train_options(["default"])

    # train and log the results
    net.train(train_set, epochs=150, test_dataset=eval_set, verbosity=2, num_workers=6, batch_size=6,
              output_policy=gem.ToConsoleAndExcelFile(excel_file_name))


summarize_sets()

#train([], [], current_field)
#train(["loss_function"], [torch.nn.HingeEmbeddingLoss], current_field)
#train(["loss_function"], [torch.nn.CosineEmbeddingLoss], current_field)
#train(["loss_function"], [torch.nn.MSELoss], current_field)

