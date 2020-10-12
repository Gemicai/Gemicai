from abc import ABC, abstractmethod


class OutputPolicy(ABC):

    @abstractmethod
    def __init__(self):
        pass


class OutputToConsole(OutputPolicy):

    def __init__(self):
        None


class OutputToExcelFile(OutputPolicy):

    def __init__(self):
        None
