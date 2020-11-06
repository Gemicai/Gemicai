import os
import torchvision.models as models
import gemicai as gem
import unittest


class TestDefaultLastLayerConfig(unittest.TestCase):
    def setUp(self):
        self.model = models.resnet18(pretrained=True)
        self.dataset = gem.DicomoDataset.get_dicomo_dataset(os.path.join("..", "examples", "gemset", "CT"),
                                                       labels=['BodyPartExamined'])
        self.classes = self.dataset.classes('BodyPartExamined')
        self.functor = gem.classifier_functors.DefaultLastLayerConfig()

    def test__call__correct_usage(self):
        self.functor(self.model, self.classes)
        self.assertIsInstance(self.model.fc, gem.classifier_functors.nn.Linear)

    def test__call__wrong_module_type(self):
        with self.assertRaises(TypeError):
            self.functor([], self.classes)

    def test__call__wrong_classes_type(self):
        with self.assertRaises(TypeError):
            self.functor(self.model, {self.classes})


if __name__ == '__main__':
    unittest.main()

