import sys

import numpy as np


def interpolate_trigger_times(trigger_sample_nums: np.ndarray,
                              deviation_interval : float = 0.05) -> np.ndarray:
    '''
    Interpolates trigger times by guessing missing trigger times, if there are
        a few dropped triggers. Assumes that very few triggers overall were dropped

    Works by estimating the trigger interval in number of samples by taking a median
        of the trigger interval, then checking the inter-trigger-interval to see if
        there are intervals that are very near to an integer > 1 multiple of the
        trigger interval

    Assumes that there are frames_per_trigger interval, maximum deviation from integer
        corresponds to frame_num_tol number of frames worth of time


    :param trigger_sample_nums: shape (n_trigger_times, ), must be integer-valued
    :return:
    '''

    # shape (n_trigger_times - 1, )
    trigger_interval_times = trigger_sample_nums[1:] - trigger_sample_nums[:-1]

    estimated_trigger_interval = np.median(trigger_interval_times)

    # contains the approximate number of intervals for every time between triggers
    # if no triggers were dropped, every value should be very near to one
    length_undropped_interval_per_interval = trigger_interval_times / estimated_trigger_interval

    does_not_need_interpolation = np.logical_and.reduce([
        length_undropped_interval_per_interval < (1.0 + deviation_interval),
        length_undropped_interval_per_interval > (1.0 - deviation_interval)
    ])

    if np.all(does_not_need_interpolation):
        # all of the trigger times are reasonable, so don't need to do anything
        return trigger_sample_nums
    else:
        # we may have to interpolate triggers here

        # verify that all of the intervals that we may have to do the interpolation for
        # correspond to near-integer multiples of the correct trigger interval
        # If the time is too far away from an integer multiple, it's not clear what to
        # do here and so we'll dump something to standard error and do nothing

        gaps_needs_interpolation = length_undropped_interval_per_interval[~does_not_need_interpolation]
        gaps_deviation_from_integer = np.abs(np.rint(gaps_needs_interpolation) - gaps_needs_interpolation)
        if np.all(gaps_deviation_from_integer < deviation_interval):

            # Announce to stderr that we're doing trigger interpolation
            print("Performing trigger interpolation", file=sys.stderr)

            # do the trigger interpolation
            prev_trigger_time = trigger_sample_nums[0]
            interpolated_trigger_times = [prev_trigger_time, ]
            for t_time, n_intervals in zip(trigger_sample_nums[1:], length_undropped_interval_per_interval):

                # if the interval is sufficiently near to 1 unit, pass it through
                # otherwise add enough triggers with the appropriate interval such that the inter-trigger interval
                # is close to 1 unit

                # round the interval fraction to the nearest integer
                rounded_int_n_intervals = int(np.rint(n_intervals))

                if rounded_int_n_intervals == 1:
                    interpolated_trigger_times.append(t_time)

                else:
                    # linspace and convert the result to integer
                    extra_trigger_times = np.linspace(prev_trigger_time, t_time,
                                                num=rounded_int_n_intervals+1, endpoint=True)[1:]
                    for x in extra_trigger_times:
                        interpolated_trigger_times.append(int(np.rint(x)))

                prev_trigger_time = t_time

            return np.array(interpolated_trigger_times)

        else:

            # Announce to stderr that we can't figure out how to do trigger interpolation
            print("Cannot do trigger interpolation, gaps between trigger times are non-integer" + \
                  "multiple of expected trigger interval",
                  file=sys.stderr)
            # Give up
            return trigger_sample_nums

