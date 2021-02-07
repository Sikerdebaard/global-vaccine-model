import numpy as np


def estimate_vaccinated_from_doses(doses_administered, interval, snoop_intervals=[-5, 5], cumulative_output=True):
    if isinstance(interval, list):
        intervals = interval
    else:
        intervals = [interval] * len(doses_administered)

    # add 1 because the interval excludes the day when the shot was given
    intervals = [x+1 for x in intervals]

    maxinterval = max(intervals) + 1

    vaccinated = list(doses_administered)
    vaccinated = vaccinated + ([9_999_999_999_999] * maxinterval)
    fully_vaccinated = [0] * len(vaccinated)
    snoop_mask = [False] * len(vaccinated)

    for i in range(0, len(vaccinated) - maxinterval):
        interval = intervals[i]
        if vaccinated[i + interval] >= vaccinated[i]:
            vaccinated[i + interval] = vaccinated[i + interval] - vaccinated[i]
            fully_vaccinated[i + interval] += vaccinated[i]
        else:  # snoop from surrounding days
            snoop_mask[i] = True
            num_doses_undistributed = vaccinated[i] - vaccinated[i + interval]
            fully_vaccinated[i + interval] += vaccinated[i] - num_doses_undistributed
            vaccinated[i + interval] = 0
            for sinterval in snoop_intervals:
                if sinterval < 0:
                    iterator = range(i - 1, i + interval - 1, -1)  # this goes from i-1 up to and including i-interval
                else:
                    iterator = range(i + 1, i + interval + 1)
                for j in iterator:
                    if vaccinated[j + interval] >= num_doses_undistributed:
                        vaccinated[j + interval] -= num_doses_undistributed
                        fully_vaccinated[j + interval] += num_doses_undistributed
                        num_doses_undistributed = 0
                        break  # exit for loop
                    else:
                        num_doses_undistributed = vaccinated[j] - vaccinated[j + interval]
                        fully_vaccinated[j + interval] += vaccinated[j + interval]  # vaccinated[j] - num_doses_undistributed
                        vaccinated[j + interval] = 0

            assert num_doses_undistributed == 0, f'Error dose distribution on day {i}, undistributed: {num_doses_undistributed}. Consider increasing snoop_intervals.'

    maxlen = len(vaccinated) - maxinterval

    if cumulative_output:
        vaccinated = np.cumsum(vaccinated[0:maxlen])
        fully_vaccinated = np.cumsum(fully_vaccinated[0:maxlen])
        single_dose = vaccinated - fully_vaccinated
    else:
        vaccinated = np.array(vaccinated[0:maxlen])
        fully_vaccinated = np.array(fully_vaccinated[0:maxlen])
        single_dose = vaccinated - fully_vaccinated

    snoop_mask = snoop_mask[0:len(vaccinated) - maxinterval]

    return vaccinated, fully_vaccinated, single_dose, snoop_mask
