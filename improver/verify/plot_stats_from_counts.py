import argparse
import os

from improver.verify.parse_file import get_model, read_count_files, set_basename
from improver.verify.statistics import StatsDict, plot_by_leadtime


def main(infiles, plotdir, stats, startdate, enddate):
    """
    Read textfiles with lines of the form:

    YYYYMMDDTHHmmZ lead_time_mins threshold_mmh hits misses false_alarms no_det

    Calculate binary statistics (POD, FAR, CSI), and plot with lead time

    Args:
        infiles (list of str):
            List of files to read
        plotdir (str):
            Full path to directory to save plots
        stats (list of str):
            List of stats to plot
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

    # TODO capability to plot different models for a single threshold, by lead time
    # Need to have plotting function take multiple stats dicts as currently designed

    start = startdate
    end = enddate
    if start is None:
        start = 0
    if end is None:
        end = 20500101

    counts_dict = read_count_files(infiles, start, end)
    stats_dict = StatsDict(counts_dict)

    for stat in stats:
        basename = set_basename(infiles, stat, startdate, enddate)
        outname = os.path.join(plotdir, basename)
        plot_by_leadtime(stats_dict, [0.03, 0.1, 0.5, 1.0, 2.0, 4.0], stat, outname)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', type=str, nargs='+', help='List of textfiles to read')
    parser.add_argument('--plotdir', type=str, help='Output directory to save plots')
    parser.add_argument('--stats', type=str, nargs='+', help='List of stats to plot')
    parser.add_argument('--startdate', type=int, default=None)
    parser.add_argument('--enddate', type=int, default=None)
    args = parser.parse_args()

    main(args.infiles, args.plotdir, args.stats, args.startdate, args.enddate)

