import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.Classifier as test
import unittest


class TestClassifier(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
