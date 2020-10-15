# All IO should be handled here, since easier to refactor code when we or someone else in the future decides that there
# is something better than pickle to create .gemset and .gemnode etc.
import pickle
import gzip
import shutil
import tempfile
import gemicai as gem

# Supported object types
supported_ojbects = [gem.Classifier, gem.ClassifierNode]

# File extestions relative to object type.
file_extensions = ['.gemclas', '.gemnode', '.gemset']

# TODO: add support for GemicaiDataset


def save(file_path, obj):
    if type(obj) not in supported_ojbects:
        raise Exception('Object of type {} is not supported by gemicai.io'.format(type(obj)))
    temp = tempfile.NamedTemporaryFile(mode="ab+", delete=False)
    pickle.dump(obj=obj, file=temp, protocol=pickle.HIGHEST_PROTOCOL)
    zip_to_file(temp, get_path_with_extension(file_path, obj))


def load(file_path, zipped=True):
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


# Returns file path with correct extesion. Dependant on object type.
def get_path_with_extension(file_path, obj):
    ext = file_extensions[supported_ojbects.index(type(obj))]
    if not file_path.endswith(ext):
        file_path += ext
    return file_path


def zip_to_file(file, zip_path):
    with gzip.open(zip_path, 'wb') as zipped:
        file = open(file.name, 'rb')
        shutil.copyfileobj(file, zipped)
        file.close()


def unzip_to_file(file, zip_path):
    with gzip.open(zip_path, 'rb') as zipped:
        file = open(file.name, 'ab+')
        shutil.copyfileobj(zipped, file)
        file.close()
