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
        subset = data.subset(("Modality", 1))
        with self.assertRaises(TypeError):
            next(iter(subset))


class TestDicomoDataset(unittest.TestCase):

    def test_from_file_correct_usage(self):
        None

    def test_from_file_wrong_file_path(self):
        None

    def test_from_file_wrong_file_type(self):
        None

    def test_from_file_wrong_label_type(self):
        None

    def test_from_file_no_labels(self):
        None

    def test_from_file_test_label_existence(self):
        None

    def test_from_file_constraints_wrong_type(self):
        None

    def test_from_file_constraints_empty(self):
        None

    def test_from_file_check_contraint(self):
        None

    def test_from_file_apply_invalid_transformation(self):
        None

    def test_from_file_apply_valid_transformation(self):
        None

    def test_from_file_correct_usage(self):
        None

    def test_from_directory_file_wrong_directory_path(self):
        None

    def test_from_directory_file_wrong_label_type(self):
        None

    def test_from_directory_file_no_labels(self):
        None

    def test_from_directory_file_test_label_existence(self):
        None

    def test_from_directory_file_constraints_wrong_type(self):
        None

    def test_from_directory_file_constraints_empty(self):
        None

    def test_from_directory_file_check_contraint(self):
        None

    def test_from_directory_file_apply_invalid_transformation(self):
        None

    def test_from_directory_file_apply_valid_transformation(self):
        None

    def test_get_dicomo_dataset_correct_usage_file(self):
        None

    def test_get_dicomo_dataset_correct_usage_directory(self):
        None


class TestPickledDicomoFilePool(unittest.TestCase):

    def test_init_correct_usage(self):
        None

    def test_init_invalid_file_pool_type(self):
        None

    def test_init_invalid_file_pool_path(self):
        None

    def test_init_file_has_wrong_type(self):
        None

    def test_init_wrong_labels_type(self):
        None

    def test_init_wrong_constraints_type(self):
        None

    def test_iter(self):
        None

    def test_next(self):
        None

    def test_len(self):
        None


class TestPickledDicomoDataFolder(unittest.TestCase):

    def test_init_correct_usage(self):
        None

    def test_init_invalid_directory_path(self):
        None

    def test_init_file_has_wrong_type(self):
        None

    def test_init_wrong_labels_type(self):
        None

    def test_init_wrong_constraints_type(self):
        None

    def test_iter(self):
        None

    def test_next(self):
        None

    def test_len(self):
        None


class TestConcurrentPickledDicomObjectTaskSplitter(unittest.TestCase):

    def test_init_correct_usage(self):
        None

    def test_init_invalid_directory_path(self):
        None

    def test_init_wrong_labels_type(self):
        None

    def test_init_wrong_constraints_type(self):
        None

    def test_iter(self):
        None

    def test_next(self):
        None

    def test_len(self):
        None

    def test_an_be_parallelized(self):
        None


if __name__ == '__main__':
    unittest.main()
