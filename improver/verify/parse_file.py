"""Module to parse textfiles created by the aggregate_counts CLI"""

import os.path as pth


def get_model(filename):
    """Get string model identifier from filename"""
    return pth.basename(filename)[7:-11]


def set_basename(infiles, stat, startdate=None, enddate=None):
    """Set output filename based on data range and statistics"""
    infiles = sorted(infiles)
    if startdate is None:
        index = pth.basename(infiles[0]).find('_')
        startdate = pth.basename(infiles[0])[:index]
    if enddate is None:
        index = pth.basename(infiles[-1]).find('_')
        enddate = pth.basename(infiles[-1])[:index]
    model = get_model(infiles[0])
    return f'{startdate}-{enddate}_{model}_{stat}.png'


def format_line(line):
    """Take raw line from textfile, remove spaces and return list of
    numbers"""
    # remove spaces
    ftline = line.strip(' \n').split(' ')
    ftline = [item for item in ftline if item != '']
    # convert to numerical values
    ftline = [ftline[0], int(ftline[1]), float(ftline[2]),
              *[int(item) for item in ftline[3:]]]
    return ftline


def read_count_files(infiles, startdate, enddate):
    """Read input counts files

    Args:
        infiles (list of str)
        startdate (int)
            Date in YYYYMMDD format, or 0
        enddate (int)
            Date in YYYYMMDD format

    Returns:
        Dictionary of hits, misses, false alarms and correct "no detections"
        by lead time and threshold
    """
    counts_by_leadtime = {}
    for datafile in infiles:
        with open(datafile) as dtf:
            line = dtf.readline()
            while line:
                dt, lt, thresh, hits, misses, false, no_det = format_line(line)
                day = int(dt[:8])
                if day > startdate and day < enddate:
                    if lt not in counts_by_leadtime:
                        counts_by_leadtime[lt] = {}
                    if thresh not in counts_by_leadtime[lt]:
                        counts_by_leadtime[lt][thresh] = {}
                        for name in ['hits', 'misses', 'false', 'no_det']:
                            counts_by_leadtime[lt][thresh][name] = 0
                    counts_by_leadtime[lt][thresh]['hits'] += hits
                    counts_by_leadtime[lt][thresh]['misses'] += misses
                    counts_by_leadtime[lt][thresh]['false'] += false
                    counts_by_leadtime[lt][thresh]['no_det'] += no_det

                line = dtf.readline()

    return counts_by_leadtime
