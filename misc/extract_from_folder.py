import subprocess
import pathlib
import string
import sys


def open_directory(path, dir_type):
    try:
        return pathlib.Path(path)
    except:
        print("Could not open a specified " + dir_type + " directory")
        sys.exit(1)


def get_file_name(path):
    try:
        return str(path).split('.')[0].split('/')[-1]
    except:
        print("Cannot parse a file path: " + str(path))
        sys.exit(1)


# ------------------------ END OF THE DECLARATIONS ------------------------

args = []

for arg in sys.argv:
    args.append(str(arg))

args = ['', 'examples', 'examples/pngs']

# Check if we got a sufficient number of parameters
try:
    if (len(args) != 3):
        raise Exception("Improper program usage")
except:
    print("Proper program usage:")
    print("python extract_from_folder.py {input_folder} {output_folder}")
    sys.exit(1)

# Open specified input directory
in_dir = open_directory(args[1], "input");

# Process files in the input directory 
for path in in_dir.iterdir():
    if path.is_file():

        # Prepare program arguments
        file_name = get_file_name(path)
        image_output = args[2] + '/images/' + file_name + '.png'
        pickle_output = args[2] + '/pickle/' + file_name + '.pkl'

        # Start a subprocess that will handle a specified file
        process = subprocess.run(
            ['python', 'extract_data.py', str(path), image_output, pickle_output],
            stdout=subprocess.PIPE,
            universal_newlines=True)

        # Check whenever subprocess terminated successfully
        if process.stdout:
            print("Could not process file: " + str(path))
            print("Reason: " + process.stdout)
        else:
            print("File " + str(path) + " processed")
