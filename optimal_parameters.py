import torchvision.models as models
import gemicai as gem
import torch
import os


dataset_all = os.path.join("examples", "gzip", "CT")
dataset_pick_middle = os.path.join("examples", "gzip", "CT")
dataset_common_modalities = os.path.join("examples", "gzip", "CT")

dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
current_field = dicom_fields[0]

excel_file_default = "default.xlsx"
excel_file_loss_function = "loss_functions.xlsx"
excel_file_optimizers = "adam_optimizer.xlsx"

train_set = gem.DicomoDataset.get_dicomo_dataset(dataset_common_modalities, labels=[current_field])
eval_set = gem.DicomoDataset.get_dicomo_dataset(dataset_pick_middle, labels=[current_field])
resnet18 = models.resnet18(pretrained=True)


def summarize_sets():
    train_set.summarize(current_field)
    eval_set.summarize(current_field)


def log_train_options(data_list, file):
    excel = gem.ToExcelFile(file)
    excel.print_row(data_list, ["A", "B", "C", "D", "E", "F"])


def train(field_list, value_list, modality, file):
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
        log_train_options(value_types, file)
    else:
        log_train_options(["default"], file)

    # train and log the results
    net.train(train_set, epochs=1, test_dataset=eval_set, verbosity=2, num_workers=6, batch_size=6,
              output_policy=gem.ToConsoleAndExcelFile(file))


#summarize_sets()

train([], [], current_field, excel_file_default)
train(["loss_function"], [torch.nn.HingeEmbeddingLoss()], current_field, excel_file_loss_function)
train(["loss_function"], [torch.nn.CosineEmbeddingLoss()], current_field, excel_file_loss_function)
#train(["loss_function"], [torch.nn.MSELoss()], current_field, excel_file_loss_function)
#train(["loss_function"], [torch.nn.SmoothL1Loss()], current_field, excel_file_loss_function)

adam = torch.optim.Adam(resnet18.parameters(), lr=0.001)
train(["loss_function", "optimizer"], [torch.nn.HingeEmbeddingLoss(), adam], current_field, excel_file_optimizers)
train(["loss_function", "optimizer"], [torch.nn.CosineEmbeddingLoss(), adam], current_field, excel_file_optimizers)
train(["loss_function", "optimizer"], [torch.nn.MSELoss(), adam], current_field, excel_file_optimizers)
train(["loss_function", "optimizer"], [torch.nn.SmoothL1Loss(), adam], current_field, excel_file_optimizers)


