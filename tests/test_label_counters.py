import os
import gemicai as gem
import unittest

raw_dicom_directory = os.path.join("..", "examples", "dicom", "CT")
raw_dicom_file_path = os.path.join(raw_dicom_directory, "325261597578315993471860132776680.dcm.gz")

dicom_directory = os.path.join("..", "examples", "gemset", "CT")
dicom_data_set = os.path.join(dicom_directory, "000001.gemset")


class TestLabelCounter(unittest.TestCase):

    def test__str__(self):
        ctr = gem.LabelCounter()
        temp_1 = str(ctr)
        ctr.update(['z'])
        temp_2 = str(ctr)
        self.assertNotEqual(temp_1, temp_2)

    def test_update_wrong_param_type(self):
        ctr = gem.LabelCounter()
        with self.assertRaises(TypeError):
            ctr.update(dict())

    def test_update_correct_param(self):
        ctr = gem.LabelCounter()
        ctr.update('z')
        self.assertEqual(ctr.dic['z'], 1)
        ctr.update('z')
        ctr.update('z')
        self.assertEqual(ctr.dic['z'], 3)
        ctr.update('m')
        self.assertEqual(ctr.dic['z'], 3)
        self.assertEqual(ctr.dic['m'], 1)

    def test_update_nested_correct_param(self):
        ctr = gem.LabelCounter()
        ctr.update([[['z']], ['k'], 'o'])
        self.assertEqual(ctr.dic['z'], 1)
        self.assertEqual(ctr.dic['k'], 1)
        self.assertEqual(ctr.dic['o'], 1)

    def test_update_on_dimcom(self):
        data = gem.load_dicom(raw_dicom_file_path)
        ctr = gem.LabelCounter()
        ctr.update(data.get('Modality'))
        self.assertEqual(ctr.dic[data.get('Modality')], 1)
        ctr.update(data.get('Modality'))
        ctr.update(data.get('Modality'))
        self.assertEqual(ctr.dic[data.get('Modality')], 3)

    def test_update_on_gemicai_dicom_object(self):
        iterator = gem.PickledDicomoDataSet(dicom_data_set, ['Modality'])
        data = next(iter(iterator))
        ctr = gem.LabelCounter()
        ctr.update(data[1])
        self.assertEqual(ctr.dic[data[1]], 1)
        ctr.update(data[1])
        ctr.update(data[1])
        self.assertEqual(ctr.dic[data[1]], 3)


if __name__ == '__main__':
    unittest.main()
