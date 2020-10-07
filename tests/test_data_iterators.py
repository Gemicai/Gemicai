import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import gemicai.data_iterators as test
import unittest


class TestDicomoDataset(unittest.TestCase):
    pass


class TestPickledDicomoDataSet(unittest.TestCase):
    pass


class TestPickledDicomoDataFolder(unittest.TestCase):
    pass


class TestPickledDicomoFilePool(unittest.TestCase):
    pass


class TestConcurrentPickledDicomoTaskSplitter(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
