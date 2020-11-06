import gemicai.dicom_utilities as test
import unittest
import pydicom
import os

wrong_dicom_file_path = os.path.join("..", "325261597578315993471860132776680.dcm.gz")

correct_dicom_directory = os.path.join("..", "examples", "dicom", "CT")
correct_dicom_file_path = os.path.join(correct_dicom_directory, "325261597578315993471860132776680.dcm.gz")

correct_dicom_object_directory = os.path.join("..", "examples", "gemset", "CT")
correct_dicom_object_file_path = os.path.join(correct_dicom_object_directory, "000001.gemset")

test_output_directory = os.path.join("..", "test_directory")


class TestLoadDicom(unittest.TestCase):

    def test_existing_file(self):
        self.assertIsInstance(test.load_dicom(correct_dicom_file_path), pydicom.dataset.FileDataset)

    def test_non_existing_file(self):
        with self.assertRaises(FileNotFoundError):
            test.load_dicom(wrong_dicom_file_path)

    def test_wrong_format(self):
        with self.assertRaises(TypeError):
            test.load_dicom(correct_dicom_object_file_path)


class TestCreateDicomObjectDatasetFromFolder(unittest.TestCase):

    def test_wrong_input(self):
        os.mkdir(test_output_directory)
        with self.assertRaises(NotADirectoryError):
            test.dicom_to_gemset(os.path.join(correct_dicom_directory, "test"),
                                 test_output_directory, ['Modality'])
        os.rmdir(test_output_directory)

    def test_wrong_output(self):
        with self.assertRaises(NotADirectoryError):
            test.dicom_to_gemset(correct_dicom_directory,
                                 test_output_directory, ['Modality'])

    def test_wrong_objects_per_file_type(self):
        os.mkdir(test_output_directory)
        with self.assertRaises(TypeError):
            test.dicom_to_gemset(correct_dicom_directory,
                                 test_output_directory, ['Modality'], objects_per_file=None)
        os.rmdir(test_output_directory)

    def test_correct_usage(self):
        os.mkdir(test_output_directory)
        try:
            test.dicom_to_gemset(correct_dicom_directory,
                                 test_output_directory, ['Modality'], objects_per_file=25)
            size = 0
            for root, dirs, files in os.walk(test_output_directory):
                for file in files:
                    os.remove(os.path.join(root, file))
                    size += 1
            self.assertEqual(size, 2)
        finally:
            os.rmdir(test_output_directory)

    def test_correct_usage_pick_middle_true(self):
        os.mkdir(test_output_directory)
        try:
            test.dicom_to_gemset(
                correct_dicom_directory, test_output_directory, ['Modality'], objects_per_file=50, pick_middle=True)
            size = 0

            for root, dirs, files in os.walk(test_output_directory):
                for file in files:
                    os.remove(os.path.join(root, file))
                    size += 1
            self.assertEqual(size, 1)
        finally:
            os.rmdir(test_output_directory)


if __name__ == '__main__':
    unittest.main()

