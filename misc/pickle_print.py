import pickle
import sys
import os

argv = sys.argv
argv = ['', 'examples/pngs/pickle/1.pkl']

try:
    if (len(argv) < 2):
        raise Exception("Specify a valid pickle file as the paramter")
    infile = open(os.path.normpath(argv[1]), 'rb')
    print(pickle.load(infile))
    infile.close()
except:
    print("Proper usage:")
    print("pickle_print.py {pickle_file_path}.pkl")
    sys.exit(1)
