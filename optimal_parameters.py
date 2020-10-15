import torchvision.models as models
import gemicai as gem
import torch
import os


eval_dataset = '/mnt/SharedStor/eval_dataset'
train_dataset = '/mnt/SharedStor/train_dataset'

dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
current_field = dicom_fields[0]

train_set = gem.DicomoDataset.get_dicomo_dataset(train_dataset, labels=[current_field])
train_set_classes = train_set.classes(current_field)

eval_set = gem.DicomoDataset.get_dicomo_dataset(eval_dataset, labels=[current_field])
eval_set_classes = eval_set.classes(current_field)


def summarize_sets():
    train_set.summarize(current_field)
    eval_set.summarize(current_field)


def log_train_options(data_list, file):
    excel = gem.ToExcelFile(file)
    excel.print_row(data_list, ["A", "B", "C", "D", "E", "F"])


def _train(model, field_list, value_list, file):
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


def _train_with_loss_function(model, excel_file, optimizer=None):

    field_list = []
    value_type = []

    _train(model, field_list, value_type, excel_file)
    if optimizer is not None:
        field_list = ["optimizer"]
        value_type = [optimizer]
        _train(model, field_list, value_type, excel_file)

    field_list = ["loss_function"]
    value_type = [torch.nn.HingeEmbeddingLoss()]
    _train(model, field_list, value_type, excel_file)

    if optimizer is not None:
        field_list += ["optimizer"]
        value_type += [optimizer]
        _train(model, field_list, value_type, excel_file)

    # TODO: TypeError: forward() missing 1 required positional argument: 'target'
    # value_type[0] = torch.nn.CosineEmbeddingLoss()
    # train(model, field_list, value_type, excel_file)

    # TODO: UserWarning: Using a target size (torch.Size([6])) that is different to the input size (torch.Size([6, 1])).
    # TODO: This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
    # value_type[0] = torch.nn.torch.nn.MSELoss()
    # train(model, field_list, value_type, excel_file)
    # value_type[0] = torch.nn.SmoothL1Loss()
    # train(model, field_list, value_type excel_file)

def train(model, excel_file):
    _train_with_loss_function(model, excel_file, None)
    # _train_with_loss_function(model, excel_file, torch.optim.Adadelta(model.parameters()))
    # _train_with_loss_function(model, excel_file, torch.optim.Adagrad(model.parameters()))
    _train_with_loss_function(model, excel_file, torch.optim.Adam(model.parameters()))
    # _train_with_loss_function(model, excel_file, torch.optim.Adamax(model.parameters()))
    # _train_with_loss_function(model, excel_file, torch.optim.ASGD(model.parameters()))
    # _train_with_loss_function(model, excel_file, torch.optim.RMSprop(model.parameters()))
    # _train_with_loss_function(model, excel_file, torch.optim.Rprop(model.parameters()))

    # TODO: RuntimeError: SparseAdam does not support dense gradients, please consider Adam instead
    # _train_with_loss_function(model, excel_file, torch.optim.SparseAdam(model.parameters()))

    # TODO: TypeError: step() missing 1 required positional argument: 'closure'
    # _train_with_loss_function(model, excel_file, torch.optim.LBFGS(model.parameters()))


excel_folder = "/mnt/SharedStor/excel_outputs/"
excel_resnet18 = os.path.join(excel_folder, "resnet18.xlsx")
excel_alexnet = os.path.join(excel_folder, "alexnet.xlsx")
excel_squeezenet = os.path.join(excel_folder, "squeezenet.xlsx")
excel_vgg16 = os.path.join(excel_folder, "vgg16.xlsx")
excel_densenet = os.path.join(excel_folder, "densenet.xlsx")
excel_inception = os.path.join(excel_folder, "inception.xlsx")
excel_googlenet = os.path.join(excel_folder, "googlenet.xlsx")
excel_shufflenet = os.path.join(excel_folder, "shufflenet.xlsx")
excel_mobilenet = os.path.join(excel_folder, "mobilenet.xlsx")
excel_resnext50_32x4d = os.path.join(excel_folder, "resnext50_32x4d.xlsx")
excel_wide_resnet50_2 = os.path.join(excel_folder, "wide_resnet50_2.xlsx")
excel_mnasnet = os.path.join(excel_folder, "mnasnet.xlsx")


summarize_sets()

# TODO: Those models work
# train(models.resnet18(pretrained=True), excel_resnet18)
# train(models.googlenet(pretrained=True), excel_googlenet)
# train(models.shufflenet_v2_x1_0(pretrained=True), excel_shufflenet)
# train(models.resnext50_32x4d(pretrained=True), excel_resnext50_32x4d)
# train(models.wide_resnet50_2(pretrained=True), excel_wide_resnet50_2)
# train(models.alexnet(pretrained=True), excel_alexnet)
# train(models.vgg16(pretrained=True), excel_vgg16)
# train(models.mobilenet_v2(pretrained=True), excel_mobilenet)
# train(models.mnasnet1_0(pretrained=True), excel_mnasnet)

# TODO: Calculated padded input size per channel: (3 x 3). Kernel size: (5 x 5).
# TODO: Kernel size can't be greater than actual input size
# train(models.inception_v3(pretrained=True), excel_inception)

# TODO: TypeError: 'Linear' object is not subscriptable
# train(models.densenet161(pretrained=True), excel_densenet)

# TODO: no in_features
# train(models.squeezenet1_0(pretrained=True), excel_squeezenet)
