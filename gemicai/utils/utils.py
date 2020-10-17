from string import Template
import os
from tabulate import tabulate
from collections import Counter


class DeltaTemplate(Template):
    delimiter = "%"


# similar to strftime, but this one is for a datetime.timedelta object.
def strfdelta(tdelta, fmt='%H:%M:%S'):
    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def get_directory_info(directory):
    total_size = 0
    cnt_files
    for root, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return str(tabulate(data, headers=['Extension', 'Files', 'Size'], tablefmt='orgtbl'))