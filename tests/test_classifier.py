import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import torchvision.models as models
import gemicai.Classifier as test
import gemicai as gem
import unittest

data_path = os.path.join("examples", "gzip", "CT")
data_set = os.path.join(data_path, "000001.gemset")
test_classifier_dir = os.path.join(parent_dir, "test_directory")

model = models.resnet18(pretrained=True)
train_dataset = gem.DicomoDataset.get_dicomo_dataset(data_path, labels=['BodyPartExamined'])
eval_dataset = gem.DicomoDataset.get_dicomo_dataset(data_path, labels=['BodyPartExamined'])


class TestClassifier(unittest.TestCase):

    def test_init_correct_usage(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        self.assertIsInstance(classifier, gem.Classifier)
        self.assertIsInstance(classifier.module, gem.nn.Module)
        self.assertIsInstance(classifier.classes, list)
        self.assertIsInstance(classifier.layer_config, gem.functr.GEMICAIABCFunctor)
        self.assertIsInstance(classifier.loss_function, gem.nn.Module)
        self.assertIsInstance(classifier.optimizer, gem.torch.optim.Optimizer)

    def test__init__wrong_layer_config_type(self):
        with self.assertRaises(TypeError):
            gem.Classifier(model, train_dataset.classes('BodyPartExamined'), layer_config=[])

    def test__init__custom_layer_config_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'),
                                    layer_config=gem.functr.DefaultLastLayerConfig())
        self.assertIsInstance(classifier.layer_config, gem.functr.DefaultLastLayerConfig)

    def test__init__wrong_loss_function_type(self):
        with self.assertRaises(TypeError):
            gem.Classifier(model, train_dataset.classes('BodyPartExamined'), loss_function=[])

    def test__init__custom_loss_function_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'),
                                    loss_function=gem.nn.MultiLabelMarginLoss())
        self.assertIsInstance(classifier.loss_function, gem.nn.MultiLabelMarginLoss)

    def test__init__wrong_optimizer_type(self):
        with self.assertRaises(TypeError):
            gem.Classifier(model, train_dataset.classes('BodyPartExamined'), optimizer=[])

    def test__init__custom_optimizer_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'),
                                    optimizer=gem.torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9))
        self.assertIsInstance(classifier.optimizer, gem.torch.optim.SGD)

    def test__init__wrong_enable_cuda_type(self):
        try:
            with self.assertRaises(TypeError):
                gem.Classifier(model, train_dataset.classes('BodyPartExamined'), enable_cuda=list())
        except RuntimeError:
            None

    def test__init__correct_device_type(self):
        try:
            classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'), enable_cuda=True)
            self.assertEqual(str(classifier.device), "cuda")
        except RuntimeError:
            None

    def test__init__wrong_cuda_device_type(self):
        try:
            with self.assertRaises(TypeError):
                gem.Classifier(model, train_dataset.classes('BodyPartExamined'), enable_cuda=True, cuda_device="6")
        except RuntimeError:
            None

    def test__init__cuda_correct_initialization(self):
        try:
            classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'),
                                        enable_cuda=True, cuda_device=0)
            self.assertEqual(str(classifier.device), "cuda:0")
        except RuntimeError:
            None

    def test_train_correct_usage(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        classifier.train(train_dataset, epochs=1, pin_memory=True, test_dataset=eval_dataset)

    def test_train_wrong_dataset_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train([], epochs=1)

    def test_train_wrong_test_dataset_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs=1, test_dataset=[])

    def test_train_wrong_batch_size_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, batch_size="z", epochs=1, pin_memory=True)

    def test_train_wrong_epochs_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs="z", pin_memory=True)

    def test_train_negative_epochs(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs=-1, pin_memory=True)

    def test_train_wrong_num_workers_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs=1, pin_memory=True, num_workers="z")

    def test_train_negative_num_workers(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs=1, pin_memory=True, num_workers=-1)

    def test_train_pin_memory_wrong_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs=1, pin_memory=None)

    def test_train_wrong_verbosity_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs=1, verbosity="z")

    def test_train_negative_verbosity_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.train(train_dataset, epochs=1, verbosity=-1)

    def test_evaluate_correct_usage(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        classifier.evaluate(eval_dataset)

    def test_evaluate_wrong_dataset_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.evaluate([])

    def test_evaluate_wrong_batch_size_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.evaluate(eval_dataset, batch_size="z")

    def test_evaluate_wrong_num_workers_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.evaluate(eval_dataset, num_workers="z")

    def test_evaluate_def_negative_num_workers(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.evaluate(eval_dataset, num_workers=-10)

    def test_evaluate_pin_memory_wrong_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.evaluate(eval_dataset, pin_memory="z")

    def test_evaluate_wrong_verbosity_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.evaluate(eval_dataset, verbosity="z")

    def test_evaluate_negative_verbosity_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.evaluate(eval_dataset, verbosity=-1)

    def test_save_correct_path(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        os.mkdir(test_classifier_dir)
        try:
            classifier_path = os.path.join(test_classifier_dir, "1.pkl")
            classifier.save(classifier_path)
            self.assertEqual(os.path.isfile(classifier_path), True)
        finally:
            try:
                os.remove(classifier_path)
            except FileNotFoundError:
                None
            os.rmdir(test_classifier_dir)

    def test_save_invalid_path(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        classifier_path = os.path.join(test_classifier_dir, "1.pkl")
        with self.assertRaises(FileNotFoundError):
            classifier.save(classifier_path)

    def test_save_wrong_path_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        classifier_path = 1
        with self.assertRaises(TypeError):
            classifier.save(classifier_path)

    def test_set_trainable_layers_layers_wrong_type(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
        with self.assertRaises(TypeError):
            classifier.set_trainable_layers({})

    def test_set_trainable_layers_empty_layers(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))

        def test_mode(classifier, layer_mode):
            classifier.set_trainable_layers([("fc", layer_mode)])

            for name, param in classifier.module.named_parameters():
                name = '.'.join(name.split('.')[:-1])
                if name == "fc":
                    self.assertEqual(param.requires_grad, layer_mode)

            classifier.set_trainable_layers([])
            for name, param in classifier.module.named_parameters():
                name = '.'.join(name.split('.')[:-1])
                if name == "fc":
                    self.assertEqual(param.requires_grad, layer_mode)

        test_mode(classifier, True)
        test_mode(classifier, False)

    def test_set_trainable_layers_correct_usage(self):
        classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))

        def test_mode(classifier, layer_mode):
            classifier.set_trainable_layers([("fc", layer_mode)])
            for name, param in classifier.module.named_parameters():
                name = '.'.join(name.split('.')[:-1])
                if name == "fc":
                    self.assertEqual(param.requires_grad, layer_mode)

        test_mode(classifier, True)
        test_mode(classifier, False)

    def test_from_file_invalid_path(self):
        classifier_path = os.path.join(test_classifier_dir, "1.pkl")
        with self.assertRaises(FileNotFoundError):
            gem.Classifier.from_file(classifier_path)

    def test_from_file_wrong_file_format(self):
        classifier_path = 1
        with self.assertRaises(TypeError):
            gem.Classifier.from_file(classifier_path)

    def test_from_file_wrong_file_format(self):
        classifier_path = data_set
        with self.assertRaises(gem.pickle.UnpicklingError):
            gem.Classifier.from_file(classifier_path)

    def test_from_file_pickled_file_but_wrong_data_inside(self):
        os.mkdir(test_classifier_dir)
        try:
            test_file_path = os.path.join(test_classifier_dir, "1.pkl")
            variable = list()
            with open(test_file_path, 'wb') as output:
                gem.pickle.dump(variable, output, gem.pickle.HIGHEST_PROTOCOL)

            self.assertEqual(os.path.isfile(test_file_path), True)
            with self.assertRaises(TypeError):
                gem.Classifier.from_file(test_file_path)
        finally:
            try:
                os.remove(test_file_path)
            except FileNotFoundError:
                None
            os.rmdir(test_classifier_dir)

    def test_from_file_correct_usage(self):
        os.mkdir(test_classifier_dir)
        try:
            self.maxDiff = None
            test_file_path = os.path.join(test_classifier_dir, "1.pkl")
            classifier = gem.Classifier(model, train_dataset.classes('BodyPartExamined'))
            classifier.save(test_file_path)
            classifier = gem.Classifier.from_file(test_file_path)
            self.assertIsInstance(classifier, gem.Classifier)
        finally:
            try:
                os.remove(test_file_path)
            except FileNotFoundError:
                None
            os.rmdir(test_classifier_dir)


if __name__ == '__main__':
    unittest.main()
