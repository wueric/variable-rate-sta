import numpy as np

def interpolate_ttls(ttl_sequence : np.ndarray,
                     ttl_expected_interval : float,
                     tolerance_pm_percent : float) -> np.ndarray:
    '''

    :param ttl_sequence: TTL trigger times, shape (n_ttl_frames, )
    :param ttl_expected_interval: Expected number of samples per TTL
    :param tolerance_pm_percent: Percent deviation from integer multiple
        of the true TTL time. If within this window, then we assume
        that a full trigger was dropped, and tha we should interpolate
    :return:
    '''

    time_delta = ttl_sequence[1:] - ttl_sequence[:-1]

    time_delta_multiple = (ttl_expected_interval / ttl_expected_interval)

    # we're only interested time_delta_multiple is greater than approximately 2
    # since these are the only cases in which trigger interpolation is plausible