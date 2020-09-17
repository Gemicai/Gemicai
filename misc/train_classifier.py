import classifier
import torchvision.models as models

# create a pickle to use as a data source
# origin = os.path.join('examples', 'dicom', 'CT')
# destination = os.path.join('examples', 'compressed', 'CT/')
# PickleDataSet.dicomo.compress_dicom_files(origin, destination)

resnet18 = models.resnet18(pretrained=True)
classifier = classifier.Classifier(resnet18)
dataloader = classifier.get_data_loader()


# fetch a new batch
for tensor, bpe in dataloader:
    # in this case each batch contains 4 tensors and labels
    # trying to feed the input to the model
    classifier.model(tensor)
    print("IT DID NOT CRASH")

    # print the tensors and labels
    # WARNING!
    # only works if PickleDataSet in the get_data_loader function does not takes transform
    # variable as a parameter, Grayscale(3) somehow fucks it up
    # PickleDataSet.print_labels_and_display_images(tensor, bpe)

# print('Classifier summaray per layer, keras style')
# classifier.summary()
# print('Classifier summaray per layer')
# print(classifier.model)
#
# classifier.set_trainable_layers('all', False)
# classifier.summary()

# pkl = 'classifiers/resnet18.pkl'
# classifier.save(pkl)
# del classifier
# classifier = load_classifier(pkl)