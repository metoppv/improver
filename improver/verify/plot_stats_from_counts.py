import argparse

from improver.verify.parse_file import get_model, read_count_files
from improver.verify.statistics import StatsDict, plot_by_leadtime


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
    model = get_model(infiles[0])
    if len(infiles) > 1:
        for fname in infiles:
            if get_model(fname) != model:
                raise ValueError('All input counts must come from the same model')

    if startdate is None:
        startdate = 0
    if enddate is None:
        enddate = 20500101

    counts = read_count_files(infiles, startdate, enddate)
    stats = StatsDict(counts)

    plot_by_leadtime(stats, [0.1, 0.5, 1.0, 2.0, 4.0], 'FAR')



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', type=str, nargs='+', help='List of textfiles to read')
    parser.add_argument('--plotdir', type=str, help='Output directory to save plots')
    parser.add_argument('--startdate', type=int, default=None)
    parser.add_argument('--enddate', type=int, default=None)
    args = parser.parse_args()

    main(args.infiles, args.plotdir, args.startdate, args.enddate)

