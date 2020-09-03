"""Module to calculate and plot binary statistics from counts"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stats(hits, misses, false, no_det):
    """Calculate binary statistics"""
    a, b, c, d = [float(x) for x in [hits, misses, false, no_det]]

    POD = a / (a + b)
    FAR = c / (a + c)
    CSI = a / (a + b + c)
    HSS = 2*(a*d - b*c) / ((a+c)*(c+d) + (a+b)*(b+d))

    return POD, FAR, CSI, HSS


class StatsDict:
    """Dictionary wrapper with accessors to simplify plotting by threshold
    and lead time"""

    def __init__(self, counts):
        """Creates a new dictionary containing binary statistics by
        leadtime and threshold"""
        self.data = {}
        for lt in counts:
            self.data[lt] = {}
            for thresh in counts[lt]:
                self.data[lt][thresh] = {}
                pod, far, csi, hss = calc_stats(
                    counts[lt][thresh]['hits'],
                    counts[lt][thresh]['misses'],
                    counts[lt][thresh]['false'],
                    counts[lt][thresh]['no_det'],
                )
                self._set_pod(lt, thresh, pod)
                self._set_far(lt, thresh, far)
                self._set_csi(lt, thresh, csi)
                self._set_hss(lt, thresh, hss)

    def _set_pod(self, lt, thresh, pod):
        self.data[lt][thresh]['POD'] = pod

    def _set_far(self, lt, thresh, far):
        self.data[lt][thresh]['FAR'] = far

    def _set_csi(self, lt, thresh, csi):
        self.data[lt][thresh]['CSI'] = csi

    def _set_hss(self, lt, thresh, hss):
        self.data[lt][thresh]['HSS'] = hss

    @staticmethod
    def _sort_by_x(x, skill):
        sorted_lists = sorted(zip(x, skill))
        x = [item for item, _ in sorted_lists]
        skill = [item for _, item in sorted_lists]    
        return x, skill    

    def trend_with_leadtime(self, thresh, stat):
        """Returns a statistic with lead time for a given threshold"""
        leadtimes = []
        skill = []
        for lt in self.data:
            leadtimes.append(lt)
            skill.append(self.data[lt][thresh][stat])
        return self._sort_by_x(leadtimes, skill)

    def trend_with_threshold(self, leadtime, stat):
        """Returns a statistic with threshold for a given lead time"""
        thresholds = []
        skill = []
        for thresh in self.data[leadtime]:
            thresholds.append(thresh)
            skill.append(self.data[leadtime][thresh][stat])
        return self._sort_by_x(thresholds, skill)


def plot_by_leadtime(stats_dict, thresholds, stat):
    """Plots statistic "stat" by lead time for a range of thresholds"""
    plt.figure(figsize=(8, 6))
    for thresh in thresholds:
        leadtime, skill = stats_dict.trend_with_leadtime(thresh, stat)
        plt.plot(leadtime, skill, label='{:.2f} mm/h'.format(thresh))
    plt.legend()
    plt.xlabel('Lead time (minutes)')
    plt.xlim(left=0)
    plt.xticks(np.arange(0, leadtime[-1]+1, 60))
    plt.ylabel(stat)
    plt.ylim(0, 1)
    plt.title(f'{stat} with lead time')

    plt.show()


