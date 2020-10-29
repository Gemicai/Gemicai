"""This module contains loosely related utility functions used by the Gemicai"""

from string import Template
import os
from tabulate import tabulate
from collections import Counter
from math import log


def strfdelta(tdelta, fmt='%H:%M:%S'):
    """Similar to strftime, but this one is for a datetime.timedelta object.

    :param tdelta: datetime object containing some time difference
    :type tdelta: datetime.timedelta
    :param fmt: string with a format
    :type fmt: str
    :return: string containing a time in a given format
    """

    class DeltaTemplate(Template):
        delimiter = "%"

    d = {"D": tdelta.days}
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d["H"] = '{:02d}'.format(hours)
    d["M"] = '{:02d}'.format(minutes)
    d["S"] = '{:02d}'.format(seconds)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def format_byte_size(num):
    """Returns a given number as a formatted binary unit string

    :param num: number to format eg. 1048576
    :type num: int
    :return: a binary unit formatted string eg. 10 MB
    """
    unit_list = list(zip(['bytes', 'kB', 'MB', 'GB', 'TB', 'PB'], [0, 0, 1, 2, 2, 2]))
    if num > 1:
        exponent = min(int(log(num, 1024)), len(unit_list) - 1)
        quotient = float(num) / 1024 ** exponent
        unit, num_decimals = unit_list[exponent]
        format_string = '{:.%sf} {}' % num_decimals
        return format_string.format(quotient, unit)
    if num == 0:
        return '0 bytes'
    if num == 1:
        return '1 byte'


def dir_info(directory):
    """Prints extensions, file names and sizes of all files contained inside of a specified directory.

    :param directory: directory to iterate over
    :type directory: Union[str, os.path]
    """
    if not os.path.isdir(directory):
        raise NotADirectoryError('{} isn\'t a directory'.format(directory))
    cnt_ext, sum_size = Counter(), {}
    for root, dirs, files in os.walk(directory):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                ext = f[f.find('.'):]
                cnt_ext.update({ext: 1})
                if ext in sum_size:
                    sum_size[ext] += os.path.getsize(fp)
                else:
                    sum_size[ext] = os.path.getsize(fp)
    data = []
    for k in sum_size:
        data.append([k, cnt_ext[k], format_byte_size(sum_size[k])])
    if len(data) > 1:
        data.append(['TOTAL', sum(cnt_ext.values()), format_byte_size(sum(sum_size.values()))])
    print(tabulate(data, headers=['Extension', 'Files', 'Size'], tablefmt='orgtbl'), '\n')


def table_print(template, data, is_header=False):
    """Prints a row of a table using a specified template and data

    :param template: list of strings containing a table row template
    :type template: list
    :param data: list of strings containing a table row data
    :type data: list
    :param is_header: whenever passed data should be formatted and printed as a header
    :type is_header: bool
    """
    assert len(template) == len(data), 'Template length and data length should be equal!'
    for i, d in enumerate(data):
        data[i] = template[i].format(str(d))
    s = '| '
    for d in data:
        s += d + ' | '
    print(s)
    if is_header:
        line = '|'
        for d in data:
            line += '-' * len(d) + '--+'
        line = line[:-1]
        print(line+'|')
    # if is_header:
    #     print(tabulate([[]], headers=data, tablefmt='orgtbl'))
    # else:
    #     print(tabulate(data, tablefmt='orgtbl'))

