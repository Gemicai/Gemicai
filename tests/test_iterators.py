import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.data_iterators as test
import torchvision
import unittest

raw_dicom_directory = os.path.join(parent_dir, "examples", "dicom", "CT")
raw_dicom_file_path = os.path.join(raw_dicom_directory, "325261597578315993471860132776680.dcm.gz")

wrong_dicom_file_path = os.path.join(parent_dir, "000001.gemset")
dicom_directory = os.path.join(parent_dir, "examples", "gzip", "CT")
dicom_data_set = os.path.join(dicom_directory, "000001.gemset")


class TestPickledDicomoDataSet(unittest.TestCase):

    def test_init_correct_usage(self):
        dataset = test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints={})
        self.assertIsInstance(dataset, test.PickledDicomoDataSet)

    def test_init_invalid_file_path(self):
        with self.assertRaises(FileNotFoundError):
            test.PickledDicomoDataSet(wrong_dicom_file_path, ["CT"], constraints={})

    def test_init_file_has_wrong_type(self):
        dataset = test.PickledDicomoDataSet(raw_dicom_file_path, ["CT"], constraints={})
        with self.assertRaises(test.gem.pickle.UnpicklingError):
            next(iter(dataset))

    def test_init_wrong_labels_type(self):
        with self.assertRaises(TypeError):
            dataset = test.PickledDicomoDataSet(dicom_data_set, {"CT"}, constraints={})

    def test_init_wrong_constraints_type(self):
        with self.assertRaises(TypeError):
            dataset = test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints=[])

    def test_iter(self):
        dataset = test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints={})
        dataset = iter(dataset)
        self.assertIsInstance(dataset, test.PickledDicomoDataSet)

    def test_next(self):
        dataset = test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints={})
        data = next(iter(dataset))
        self.assertIsInstance(data, list)

    def test_len(self):
        dataset = iter(test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints={}))
        self.assertEqual(len(dataset), 0)
        next(dataset)
        self.assertEqual(len(dataset), 1)
        next(dataset)
        next(dataset)
        self.assertEqual(len(dataset), 3)

    def test_from_file_apply_invalid_transformation(self):
        with self.assertRaises(Exception):
            next(iter(test.PickledDicomoDataSet(dicom_data_set, ["CT"], transform=[], constraints={})))

    def test_from_file_apply_valid_transformation(self):
        t1 = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((244, 244)),
            torchvision.transforms.ToTensor()
        ])

        data = next(iter(test.PickledDicomoDataSet(dicom_data_set, ["CT"], transform=t1, constraints={})))
        self.assertIsInstance(data, list)

    def test_subset_correct_usage(self):
        data = test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints={})
        subset = data.subset({"Modality": "asd"})
        with self.assertRaises(StopIteration):
            next(iter(subset))

    def test_subset_wrong_constraint_type(self):
        data = test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints={})
        with self.assertRaises(TypeError):
            subset = data.subset(("Modality", 1))

    def test_can_be_parallelized(self):
        data = test.PickledDicomoDataSet(dicom_data_set, ["CT"], constraints={})
        self.assertEqual(data.can_be_parallelized(), False)


class TestPickledDicomoDataFolder(unittest.TestCase):

    def test_init_correct_usage(self):
        dataset = test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={})
        self.assertIsInstance(dataset, test.PickledDicomoDataFolder)

    def test_init_invalid_directory_path(self):
        with self.assertRaises(NotADirectoryError):
            test.PickledDicomoDataFolder(os.path.join(dicom_directory, "asd"), ["CT"], constraints={})

    def test_init_file_has_wrong_type(self):
        dataset = test.PickledDicomoDataFolder(raw_dicom_directory, ["CT"], constraints={})
        with self.assertRaises(test.gem.pickle.UnpicklingError):
            next(iter(dataset))

    def test_init_wrong_labels_type(self):
        with self.assertRaises(TypeError):
            dataset = test.PickledDicomoDataFolder(dicom_directory, {"CT"}, constraints={})

    def test_init_wrong_constraints_type(self):
        with self.assertRaises(TypeError):
            dataset = test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints=[])

    def test_iter(self):
        dataset = test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={})
        dataset = iter(dataset)
        self.assertIsInstance(dataset, test.PickledDicomoDataFolder)

    def test_next(self):
        dataset = test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={})
        data = next(iter(dataset))
        self.assertIsInstance(data, list)

    def test_len(self):
        dataset = iter(test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={}))
        self.assertEqual(len(dataset), 0)
        next(dataset)
        self.assertEqual(len(dataset), 1)
        next(dataset)
        next(dataset)
        self.assertEqual(len(dataset), 3)

    def test_subset_correct_usage(self):
        data = test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={})
        subset = data.subset({"Modality": "asd"})
        with self.assertRaises(StopIteration):
            next(iter(subset))

    def test_subset_wrong_constraint_type(self):
        data = test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={})
        with self.assertRaises(TypeError):
            subset = data.subset(("Modality", 1))

    def test_iterate_over_all(self):
        data = iter(test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={}))
        with self.assertRaises(StopIteration):
            while True:
                next(data)
        self.assertEqual(len(data), 149)

    def test_can_be_parallelized(self):
        data = test.PickledDicomoDataFolder(dicom_directory, ["CT"], constraints={})
        self.assertEqual(data.can_be_parallelized(), False)


class TestPickledDicomoFilePool(unittest.TestCase):

    def test_init_correct_usage(self):
        data = test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={})
        self.assertIsInstance(data, test.PickledDicomoFilePool)

    def test_init_invalid_file_pool_path(self):
        with self.assertRaises(FileNotFoundError):
            test.PickledDicomoFilePool([os.path.join(dicom_directory, "asd", "000001.gemset")], ["CT"], constraints={})

    def test_init_file_has_wrong_type(self):
        with self.assertRaises(test.gem.pickle.UnpicklingError):
            next(iter(test.PickledDicomoFilePool([raw_dicom_file_path], ["CT"], constraints={})))

    def test_init_wrong_labels_type(self):
        with self.assertRaises(TypeError):
            dataset = test.PickledDicomoFilePool([dicom_data_set], {"CT"}, constraints={})

    def test_init_wrong_constraints_type(self):
        with self.assertRaises(TypeError):
            dataset = test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints=[])

    def test_iter(self):
        dataset = test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={})
        dataset = iter(dataset)
        self.assertIsInstance(dataset, test.PickledDicomoFilePool)

    def test_next(self):
        dataset = test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={})
        data = next(iter(dataset))
        self.assertIsInstance(data, list)

    def test_len(self):
        dataset = iter(test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={}))
        self.assertEqual(len(dataset), 0)
        next(dataset)
        self.assertEqual(len(dataset), 1)
        next(dataset)
        next(dataset)
        self.assertEqual(len(dataset), 3)

    def test_subset_correct_usage(self):
        data = test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={})
        subset = data.subset({"Modality": "asd"})
        with self.assertRaises(StopIteration):
            next(iter(subset))

    def test_subset_wrong_constraint_type(self):
        data = test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={})
        with self.assertRaises(TypeError):
            subset = data.subset(("Modality", 1))

    def test_iterate_over_all(self):
        data = iter(test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={}))
        with self.assertRaises(StopIteration):
            while True:
                next(data)
        self.assertNotEqual(len(data), 0)

    def test_can_be_parallelized(self):
        data = test.PickledDicomoFilePool([dicom_data_set], ["CT"], constraints={})
        self.assertEqual(data.can_be_parallelized(), False)


class TestConcurrentPickledDicomObjectTaskSplitter(unittest.TestCase):

    def test_init_correct_usage(self):
        data = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={})
        self.assertIsInstance(data, test.ConcurrentPickledDicomObjectTaskSplitter)

    def test_init_invalid_directory_path(self):
        test.ConcurrentPickledDicomObjectTaskSplitter(os.path.join(dicom_directory, "asd"), ["CT"], constraints={})

    def test_init_wrong_labels_type(self):
        with self.assertRaises(TypeError):
            dataset = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, {"CT"}, constraints={})

    def test_init_wrong_constraints_type(self):
        with self.assertRaises(TypeError):
            dataset = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints=[])

    def test_iter(self):
        dataset = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={})
        dataset = iter(dataset)
        self.assertIsInstance(dataset, test.PickledDicomoFilePool)

    def test_next(self):
        dataset = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={})
        data = next(iter(dataset))
        self.assertIsInstance(data, list)

    def test_len(self):
        dataset = iter(test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={}))
        self.assertEqual(len(dataset), 0)
        next(dataset)
        self.assertEqual(len(dataset), 1)
        next(dataset)
        next(dataset)
        self.assertEqual(len(dataset), 3)

    def test_subset_correct_usage(self):
        data = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={})
        subset = data.subset({"Modality": "asd"})
        with self.assertRaises(StopIteration):
            next(iter(subset))

    def test_subset_wrong_constraint_type(self):
        data = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={})
        with self.assertRaises(TypeError):
            subset = data.subset(("Modality", 1))

    def test_iterate_over_all(self):
        data = iter(test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={}))
        with self.assertRaises(StopIteration):
            while True:
                next(data)
        self.assertNotEqual(len(data), 0)

    def test_can_be_parallelized(self):
        data = test.ConcurrentPickledDicomObjectTaskSplitter(dicom_directory, ["CT"], constraints={})
        self.assertEqual(data.can_be_parallelized(), True)


class TestDicomoDataset(unittest.TestCase):

    def test_from_file_correct_usage(self):
        dataset = test.DicomoDataset.from_file(dicom_data_set, ["CT"])
        self.assertIsInstance(dataset, test.PickledDicomoDataSet)

    def test_from_file_wrong_file_path(self):
        with self.assertRaises(FileNotFoundError):
            test.DicomoDataset.from_file(wrong_dicom_file_path, ["CT"])

    def test_from_directory_correct_usage(self):
        dataset = test.DicomoDataset.from_directory(dicom_directory, ["CT"])
        self.assertIsInstance(dataset, test.ConcurrentPickledDicomObjectTaskSplitter)

    def test_from_directory_file_wrong_directory_path(self):
        with self.assertRaises(NotADirectoryError):
            test.DicomoDataset.from_directory(os.path.join(dicom_directory, "asd"), ["CT"])

    def test_get_dicomo_dataset_correct_usage_file(self):
        dataset = test.DicomoDataset.get_dicomo_dataset(dicom_data_set)
        self.assertIsInstance(dataset, test.PickledDicomoDataSet)

    def test_get_dicomo_dataset_correct_usage_directory(self):
        dataset = test.DicomoDataset.get_dicomo_dataset(dicom_directory)
        self.assertIsInstance(dataset, test.ConcurrentPickledDicomObjectTaskSplitter)

    def test_get_dicomo_dataset_wrong_directory_path(self):
        with self.assertRaises(NotADirectoryError):
            test.DicomoDataset.get_dicomo_dataset(wrong_dicom_file_path)


if __name__ == '__main__':
    unittest.main()
