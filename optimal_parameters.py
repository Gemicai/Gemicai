import torchvision.models as models
import gemicai as gem
import torch
import os


dataset_all = os.path.join("examples", "gzip", "CT")
dataset_pick_middle = os.path.join("examples", "gzip", "CT")
dataset_common_modalities = os.path.join("examples", "gzip", "CT")

dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
current_field = dicom_fields[0]

train_set = gem.DicomoDataset.get_dicomo_dataset(dataset_common_modalities, labels=[current_field])
train_set_classes = train_set.classes(current_field)

eval_set = gem.DicomoDataset.get_dicomo_dataset(dataset_pick_middle, labels=[current_field])
eval_set_classes = eval_set.classes(current_field)


def summarize_sets():
    train_set.summarize(current_field)
    eval_set.summarize(current_field)


def log_train_options(data_list, file):
    excel = gem.ToExcelFile(file)
    excel.print_row(data_list, ["A", "B", "C", "D", "E", "F"])


def train(model, field_list, value_list, file):
    net = gem.Classifier(model, train_set_classes, enable_cuda=True)

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
    net.train(train_set, epochs=150, test_dataset=eval_set, verbosity=2, num_workers=4, batch_size=4,
              output_policy=gem.ToConsoleAndExcelFile(file))


def train_with_model(model, excel_file):
    adam = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, [], [], excel_file)
    train(model, ["loss_function"], [torch.nn.HingeEmbeddingLoss()], excel_file)

    # TODO: TypeError: forward() missing 1 required positional argument: 'target'
    # train(model, ["loss_function"], [torch.nn.CosineEmbeddingLoss()], excel_file)

    # TODO: UserWarning: Using a target size (torch.Size([6])) that is different to the input size (torch.Size([6, 1])).
    # TODO: This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
    # train(model, ["loss_function"], [torch.nn.MSELoss()], excel_file)
    # train(model, ["loss_function"], [torch.nn.SmoothL1Loss()], excel_file)

    train(model, ["optimizer"], [adam], excel_file)
    train(model, ["loss_function", "optimizer"], [torch.nn.HingeEmbeddingLoss(), adam], excel_file)

    # TODO: TypeError: forward() missing 1 required positional argument: 'target'
    # train(model, ["loss_function", "optimizer"], [torch.nn.CosineEmbeddingLoss(), adam], excel_file)

    # TODO: UserWarning: Using a target size (torch.Size([6])) that is different to the input size (torch.Size([6, 1])).
    # TODO: This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
    # train(model, ["loss_function", "optimizer"], [torch.nn.MSELoss(), adam], excel_file)
    # train(model, ["loss_function", "optimizer"], [torch.nn.SmoothL1Loss(), adam], excel_file)


excel_resnet18 = "excel_outputs/resnet18.xlsx"
excel_alexnet = "excel_outputs/alexnet.xlsx"
excel_squeezenet = "excel_outputs/squeezenet.xlsx"
excel_vgg16 = "excel_outputs/vgg16.xlsx"
excel_densenet = "excel_outputs/densenet.xlsx"
excel_inception = "excel_outputs/inception.xlsx"
excel_googlenet = "excel_outputs/googlenet.xlsx"
excel_shufflenet = "excel_outputs/shufflenet.xlsx"
excel_mobilenet = "excel_outputs/mobilenet.xlsx"
excel_resnext50_32x4d = "excel_outputs/resnext50_32x4d.xlsx"
excel_wide_resnet50_2 = "excel_outputs/wide_resnet50_2.xlsx"
excel_mnasnet = "excel_outputs/mnasnet.xlsx"


summarize_sets()

# Those models work
train_with_model(models.resnet18(pretrained=True), excel_resnet18)
train_with_model(models.googlenet(pretrained=True), excel_googlenet)
train_with_model(models.shufflenet_v2_x1_0(pretrained=True), excel_shufflenet)
train_with_model(models.resnext50_32x4d(pretrained=True), excel_resnext50_32x4d)
train_with_model(models.wide_resnet50_2(pretrained=True), excel_wide_resnet50_2)

# Those have no fc layer
# train_with_model(models.alexnet(pretrained=True), excel_alexnet)
# train_with_model(models.squeezenet1_0(pretrained=True), excel_squeezenet)
# train_with_model(models.vgg16(pretrained=True), excel_vgg16)
# train_with_model(models.densenet161(pretrained=True), excel_densenet)
# train_with_model(models.mobilenet_v2(pretrained=True), excel_mobilenet)
# train_with_model(models.mnasnet1_0(pretrained=True), excel_mnasnet)

# Calculated padded input size per channel: (3 x 3). Kernel size: (5 x 5).
# Kernel size can't be greater than actual input size
# train_with_model(models.inception_v3(pretrained=True), excel_inception)
