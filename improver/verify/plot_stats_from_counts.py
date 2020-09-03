import argparse
import os

import numpy as np
from matplotlib import pyplot as plt


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


def get_stats(hits, misses, false, no_det):
    """Calculate binary statistics"""
    a, b, c, d = [float(x) for x in [hits, misses, false, no_det]]

    POD = a / (a + b)
    FAR = c / (a + c)
    CSI = a / (a + b + c)
    HSS = 2*(a*d - b*c) / ((a+c)*(c+d) + (a+b)*(b+d))

    return POD, FAR, CSI, HSS


def main(infiles, plotdir, startdate, enddate):
    """
    Read textfiles with lines of the form:

    YYYYMMDDTHHmmZ lead_time_mins threshold_mmh hits misses false_alarms no_det

    Calculate binary statistics (POD, FAR, CSI), and plot with lead time

    Args:
        infiles (list of str):
            List of files to read
        plotdir (str):
            Full path to directory to save plots
        startdate (int or None):
            Date to start calculation in YYYYMMDD format
        enddate (int or None):
            Date to end calculation in YYYYMMDD format
    """
    # get model from first filename in list
    model = os.path.basename(infiles[0])[7:-11]
    print(model)

    if startdate is None:
        startdate = 0
    if enddate is None:
        enddate = 20500101

    counts_by_leadtime = {}
    for datafile in infiles:
        with open(datafile) as dtf:
            line = dtf.readline()
            while line:
                dt, lt, thresh, hits, misses, false, no_det = format_line(line)
                day = int(dt[:8])
                # TODO filter by threshold!
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

    for lt in counts_by_leadtime:
        for thresh in counts_by_leadtime[lt]:
            print(lt, thresh, counts_by_leadtime[lt][thresh])



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', type=str, nargs='+', help='List of textfiles to read')
    parser.add_argument('--plotdir', type=str, help='Output directory to save plots')
    parser.add_argument('--startdate', type=int, default=None)
    parser.add_argument('--enddate', type=int, default=None)
    args = parser.parse_args()

    main(args.infiles, args.plotdir, args.startdate, args.enddate)

