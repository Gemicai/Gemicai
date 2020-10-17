from string import Template
import os
from tabulate import tabulate
from collections import Counter
from math import log


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


def format_byte_size(num):
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
