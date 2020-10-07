import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.data_iterators as test
import unittest


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


class TestPickledDicomoDataSet(unittest.TestCase):

    def test_init_correct_usage(self):
        None

    def test_init_invalid_file_path(self):
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

    def test_from_file_apply_invalid_transformation(self):
        None

    def test_from_file_apply_valid_transformation(self):
        None

    def test_subset_correct_usage(self):
        None

    def test_subset_wrong_constraint_type(self):
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
