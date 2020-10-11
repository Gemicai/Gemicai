import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.dicom_utilities as test
import unittest
import pydicom
import os

wrong_dicom_file_path = os.path.join(parent_dir, "325261597578315993471860132776680.dcm.gz")

correct_dicom_directory = os.path.join(parent_dir, "examples", "dicom", "CT")
correct_dicom_file_path = os.path.join(correct_dicom_directory, "325261597578315993471860132776680.dcm.gz")

correct_dicom_object_directory = os.path.join(parent_dir, "examples", "gzip", "CT")
correct_dicom_object_file_path = os.path.join(correct_dicom_object_directory, "000001.gemset")

test_output_directory = os.path.join(parent_dir, "test_directory")


class TestLoadDicom(unittest.TestCase):

    def test_existing_file(self):
        self.assertIsInstance(test.load_dicom(correct_dicom_file_path), pydicom.dataset.FileDataset)

    def test_non_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            test.load_dicom(wrong_dicom_file_path)

    def test_wrong_format(self):
        with self.assertRaises(TypeError):
            test.load_dicom(correct_dicom_object_file_path)


class TestDicomGetTensorAndLabel(unittest.TestCase):

    def test_existing_file(self):
        self.assertIsInstance(test.dicom_get_tensor_and_label(correct_dicom_file_path), tuple)

    def test_non_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            test.dicom_get_tensor_and_label(wrong_dicom_file_path)

    def test_wrong_file_format(self):
        with self.assertRaises(TypeError):
            test.dicom_get_tensor_and_label(correct_dicom_object_file_path)


class TestCreateDicomObjectDatasetFromFolder(unittest.TestCase):

    def test_wrong_input(self):
        os.mkdir(test_output_directory)
        with self.assertRaises(NotADirectoryError):
            test.create_dicomobject_dataset_from_folder(os.path.join(correct_dicom_directory, "test"),
                                                        test_output_directory, ['Modality'])
        os.rmdir(test_output_directory)

    def test_wrong_output(self):
        with self.assertRaises(NotADirectoryError):
            test.create_dicomobject_dataset_from_folder(correct_dicom_directory,
                                                        test_output_directory, ['Modality'])

    def test_wrong_objects_per_file_type(self):
        os.mkdir(test_output_directory)
        with self.assertRaises(TypeError):
            test.create_dicomobject_dataset_from_folder(correct_dicom_directory,
                                                        test_output_directory, ['Modality'], objects_per_file=None)
        os.rmdir(test_output_directory)

    def test_correct_usage(self):
        os.mkdir(test_output_directory)
        test.create_dicomobject_dataset_from_folder(correct_dicom_directory,
                                                    test_output_directory, ['Modality'], objects_per_file=50)
        size = 0
        for root, dirs, files in os.walk(test_output_directory):
            for file in files:
                os.remove(os.path.join(root, file))
                size += 1
        os.rmdir(test_output_directory)
        self.assertEqual(size, 3)


if __name__ == '__main__':
    unittest.main()

