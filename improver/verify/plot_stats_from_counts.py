import argparse
import os

from improver.verify.parse_file import get_model, read_count_files, set_basename
from improver.verify.statistics import StatsDict, plot_by_leadtime, plot_by_threshold


def main(infiles, plotdir, stats, thresholds, startdate, enddate):
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
        thresholds (list of float):
            List of thresholds to plot
        startdate (int or None):
            Date to start calculation in YYYYMMDD format
        enddate (int or None):
            Date to end calculation in YYYYMMDD format
    """
    # set start and end times for item filtering
    start = startdate
    end = enddate
    if start is None:
        start = 0
    if end is None:
        end = 20500101

    # sort input files by model
    file_lists = {}
    for name in infiles:
        model = get_model(name)
        if model in file_lists:
            file_lists[model].append(name)
        else:
            file_lists[model] = [name]

    stats_dicts = {}
    for name in file_lists:
        counts_dict = read_count_files(file_lists[name], start, end)
        stats_dicts[name] = StatsDict(counts_dict)

    for stat in stats:
        for threshold in thresholds:
            basename = set_basename(
                infiles, stat, thresh=threshold, single_model=False,
                startdate=startdate, enddate=enddate
            )
            outname = os.path.join(plotdir, basename)
            plot_by_threshold(stats_dicts, stat, threshold, outname)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', type=str, nargs='+', help='List of textfiles to read')
    parser.add_argument('--plotdir', type=str, help='Output directory to save plots')
    parser.add_argument('--stats', type=str, nargs='+', default=['HSS', 'CSI'])
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.03, 0.1, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument('--startdate', type=int, default=None)
    parser.add_argument('--enddate', type=int, default=None)
    args = parser.parse_args()

    main(args.infiles, args.plotdir, args.stats, args.thresholds,
         args.startdate, args.enddate)

