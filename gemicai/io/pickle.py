"""IO module is used in order to interface with a file system. Entirety of this module is meant to be swapped out if
superior tools or libraries arise in the future or more performance is needed."""

# All IO should be handled here, since easier to refactor code when we or someone else in the future decides that there
# is something better than pickle to create .gemset and .gemnode etc.
import pickle
import gzip
import shutil
import tempfile
import os
import gemicai as gem

# Supported object types
supported_ojbects = [gem.Classifier, gem.ClassifierNode]

# File extestions relative to object type.
file_extensions = ['.gemclas', '.gemnode', '.gemset']

# TODO: add support for GemicaiDataset
# FIXME: zipped=True seems to not be working on linux


def save(file_path, obj, zipped=False):
    """This function saves a specified object in a given file.

    :param file_path: valid path to a file. File itself does not have to exist.
    :type file_path: str
    :param obj: object to save
    :type obj: Union[gem.Classifier, gem.ClassifierNode]
    :param zipped: whenever the file should be zipped
    :type zipped: bool
    :return:
    """
    if type(obj) not in supported_ojbects:
        raise Exception('Object of type {} is not supported by gemicai.io'.format(type(obj)))
    if zipped:
        temp = tempfile.NamedTemporaryFile(mode="ab+", delete=False)
        pickle.dump(obj=obj, file=temp, protocol=pickle.HIGHEST_PROTOCOL)
        zip_to_file(temp, get_path_with_extension(file_path, obj))
        temp.close()
        os.remove(temp.name)
    else:
        f = open(get_path_with_extension(file_path, obj), 'wb')
        pickle.dump(obj=obj, file=f, protocol=pickle.HIGHEST_PROTOCOL)


def load(file_path, zipped=False):
    """This function is used to load back a Gemicai object from the file system.

    :param file_path: valid path to a file
    :type file_path: str
    :param zipped: whenever the specified file is zipped
    :type zipped: bool
    :return: a valid Gemicai object
    :raises Exception: raised if object could not have been loaded in
    """
    if zipped:
        with gzip.open(file_path, 'rb') as inp:
            res = pickle.load(inp)
    else:
        with open(file_path, 'rb') as inp:
            res = pickle.load(inp)
    res_type = supported_ojbects[file_extensions.index('.'+file_path.split('.')[-1])]
    if type(res) != res_type:
        raise Exception('{} should contain object of type {}, instead got {}'.format(file_path, res_type, type(res)))
    return res


def get_path_with_extension(file_path, obj):
    """Returns file path with a correct extension. Dependent on the object type.

    :param file_path: path to a file without it's extension eg. /home/test/test_file. If correct extension is present
        path will be changed
    :type file_path: str
    :param obj: object which extension type we want to return
    :type obj: Union[gem.Classifier, gem.ClassifierNode]
    :return:
    """
    ext = file_extensions[supported_ojbects.index(type(obj))]
    if not file_path.endswith(ext):
        file_path += ext
    return file_path


def zip_to_file(file, zip_path):
    """Copies a content of a temporary file to the other and zips it.

    :param file: temporary file which data will be zipped
    :type file: tempfile._TemporaryFileWrapper
    :param zip_path: a valid path to a zip file, file itself does not have to exist
    :type zip_path: str
    """
    with gzip.open(zip_path, 'wb') as zipped:
        file = open(file.name, 'rb')
        shutil.copyfileobj(file, zipped)
        file.close()


def unzip_to_file(file, zip_path):
    """Unzips content of a file and copies it to the temporary file.

    :param file: temporary file which should receive zipped file's data
    :type file: tempfile._TemporaryFileWrapper
    :param zip_path: a valid path to a zip file
    :type zip_path: str
    """
    with gzip.open(zip_path, 'rb') as zipped:
        file = open(file.name, 'ab+')
        shutil.copyfileobj(zipped, file)
        file.close()
