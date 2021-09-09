import numpy as np
import torch

from whitenoise import RandomNoiseFrameGenerator

import tqdm

from typing import Dict, Tuple, Union, Sequence


def torch_single_spike_bin_select_matrix_piece(spike_time_vector: np.ndarray,
                                               n_bins_depth: int,
                                               bin_interval_samples: Union[float, int],
                                               frame_cutoff_times: np.ndarray,
                                               device: torch.device) -> torch.Tensor:
    '''

    :param spike_time: time of spike, in units of electrical samples
    :param n_bins_depth: number of bins in the STA
    :param bin_interval_samples: size of bin, in units of electrical samples
    :param frame_cutoff_times: cutoff times of each frame, including endpoints. units of electrical samples
    :return: matrix of weights corresponding, shape (n_frames, n_bins_depth)
    '''

    sum_area = (frame_cutoff_times[1:] - frame_cutoff_times[:-1]) + bin_interval_samples

    # shape (n_frames, )
    sum_area_torch = torch.tensor(sum_area, dtype=torch.float32, device=device)

    bin_backwards_times = -np.r_[n_bins_depth:-1:-1] * bin_interval_samples

    # shape (n_frames + 1, )
    frame_cutoff_times_torch = torch.tensor(frame_cutoff_times, dtype=torch.float32, device=device)

    # shape (n_spikes, )
    spike_time_vector_torch = torch.tensor(spike_time_vector, dtype=torch.float32, device=device)

    # shape (n_bins_depth + 1, )
    bin_backwards_times_torch = torch.tensor(bin_backwards_times, dtype=torch.float32, device=device)

    # shape (n_spikes, n_sta_bins + 1)
    spike_bin_times = spike_time_vector_torch[:, None] + bin_backwards_times_torch[None, :]

    # shape (n_spikes, n_frames, n_sta_bins)
    distance_to_frame_bin_end = (frame_cutoff_times_torch[None, None, 1:] - spike_bin_times[:, :-1, None]) > 0.0
    distance_to_frame_bin_begin = (spike_bin_times[:, 1:, None] - frame_cutoff_times_torch[None, None, :-1]) > 0.0

    # shape (n_spikes, n_frames, n_sta_bins)
    does_overlap = (distance_to_frame_bin_end & distance_to_frame_bin_begin)

    # shape (n_spikes, n_sta_bins, n_frames)
    upper_endpoint_maximum = torch.max(frame_cutoff_times_torch[None, None, 1:], spike_bin_times[:, 1:, None])
    lower_endpoint_minimum = torch.min(frame_cutoff_times_torch[None, None, :-1], spike_bin_times[:, :-1, None])

    # shape (n_spikes, n_sta_bins, n_frames)
    intersection_area = sum_area_torch[None, None, :] - (upper_endpoint_maximum - lower_endpoint_minimum)
    intersection_area[torch.logical_not(does_overlap)] = 0.0

    # shape (n_sta_bins, n_frames)
    return torch.sum(intersection_area, dim=0)


def torch_pack_sta_bin_select_matrix(spikes_by_cell_id: Dict[int, np.ndarray],
                                     cell_idx_offset: Dict[int, int],
                                     cell_order: Sequence[int],
                                     n_bins_depth: int,
                                     bin_interval_samples: Union[int, float],
                                     frame_cutoff_times: np.ndarray,
                                     device: torch.device) -> Tuple[torch.Tensor, Dict[int, int]]:
    '''

    :param spikes_by_cell_id: Dict, (cell_id) -> (spike time vector)
    :param cell_idx_offset: Dict, (cell_id) -> (first not-yet-looked-at index in the spike time vector)
    :param cell_order: sequence of cell_id, corresponding to the order of cells for batched matrix multiply
    :param n_bins_depth: depth of STA, in bins
    :param bin_interval_samples: width of bins, units are recorded electrical samples
    :param frame_cutoff_times: cutoff times for the STA frames, including both endpoints. units are recorded
        electrical samples
    :return:
    '''

    n_cells = len(cell_order)
    n_frames = frame_cutoff_times.shape[0] - 1

    sta_length_in_electrical_samples = np.ceil(n_bins_depth * bin_interval_samples)  # not sure what the correct
    # way to go to integer here is

    earliest_relevant_spike_sample = frame_cutoff_times[0] - sta_length_in_electrical_samples
    last_relevant_spike_sample = sta_length_in_electrical_samples + frame_cutoff_times[-1]

    bin_select_matrix = torch.zeros((n_cells, n_bins_depth, n_frames), dtype=torch.float32, device=device)

    cell_idx_offset_post = {}  # type: Dict[int, int]
    for batch_idx, cell_id in enumerate(cell_order):

        spike_times_vector = spikes_by_cell_id[cell_id]
        last_looked_at = cell_idx_offset[cell_id]

        while last_looked_at < spike_times_vector.shape[0] and \
                spike_times_vector[last_looked_at] < earliest_relevant_spike_sample:
            last_looked_at += 1

        # now we are either in relevant spike territory, or we have no spikes for this bin
        look_at_start = last_looked_at
        while last_looked_at < spike_times_vector.shape[0] and \
                spike_times_vector[last_looked_at] < last_relevant_spike_sample:
            last_looked_at += 1

        if (last_looked_at - look_at_start) > 0:
            spike_time_subvector = spike_times_vector[look_at_start:last_looked_at]

            weights_mat_single_spike = torch_single_spike_bin_select_matrix_piece(
                spike_time_subvector,
                n_bins_depth,
                bin_interval_samples,
                frame_cutoff_times,
                device
            )

            bin_select_matrix[batch_idx, ...] += weights_mat_single_spike

        to_include = cell_idx_offset[cell_id]
        while to_include < spike_times_vector.shape[0] and \
                spike_times_vector[to_include] < frame_cutoff_times[-1]:
            to_include += 1

        cell_idx_offset_post[cell_id] = to_include

    return bin_select_matrix, cell_idx_offset_post


def torch_parallel_spike_bin_select_matrix_piece(spike_time_vector: np.ndarray,
                                                 reduction_matrix: np.ndarray,
                                                 n_bins_depth: int,
                                                 bin_interval_samples: Union[float, int],
                                                 frame_cutoff_times: np.ndarray,
                                                 device: torch.device) -> torch.Tensor:
    '''
    Core engine of the bin-by-sample STA calculation

    Conventions:
        * D is the STA depth
        * Let

    Description of the algorithm: the algorithm is a parallel version of the following

    For every spike that occurs,
        (1) Generate STA frame cutoff times, corresponding to the cutoff times for each frame in the
            final computed STA (counting backwards from the spike time)
        (2) Compute the overlap (in terms of number of electrical samples) between the intervals defined
            by the STA bin cutoff times and the intervals defined by the frame transition times. These
            overlaps correspond to the weight that placed on each frame of the stimulus for the given STA
            bin and the given cell.

    Then do a matrix multiplication to produce a tensor that can be used to map the stimulus frames
        into the appropriate STA bins for every cell

    :param spike_time_vector: spike time vector for all cells under consideration,
        in units of electrical samples, shape (n_all_spikes, )
    :param reduction_matrix: matrix to multiply to replace final summation, shape (n_cells, n_all_spikes)
    :param n_bins_depth: number of bins in the STA
    :param bin_interval_samples: size of bin, in units of electrical samples
    :param frame_cutoff_times: cutoff times of each frame, including endpoints. units of electrical samples
    :return: matrix of weights corresponding, shape (n_cells, n_sta_bins, n_stimulus_frames)
    '''

    # number of electrical samples in each bin interval
    # shape (n_frames, )
    sum_area = (frame_cutoff_times[1:] - frame_cutoff_times[:-1]) + bin_interval_samples
    sum_area_torch = torch.tensor(sum_area, dtype=torch.float32, device=device)

    # shape (n_bins_depth + 1, )
    bin_backwards_times = -np.r_[n_bins_depth:-1:-1] * bin_interval_samples
    bin_backwards_times_torch = torch.tensor(bin_backwards_times, dtype=torch.float32, device=device)

    # shape (n_frames + 1, )
    frame_cutoff_times_torch = torch.tensor(frame_cutoff_times, dtype=torch.float32, device=device)

    # shape (n_all_spikes, )
    spike_time_vector_torch = torch.tensor(spike_time_vector, dtype=torch.float32, device=device)

    ###### Determine the bin in which the spike occurred ###################################
    # shape (n_all_spikes, n_sta_bins + 1)
    spike_bin_times = spike_time_vector_torch[:, None] + bin_backwards_times_torch[None, :]

    # shape (n_all_spikes, n_sta_bins, n_frames)
    upper_endpoint_maximum = torch.max(frame_cutoff_times_torch[None, None, 1:], spike_bin_times[:, 1:, None])
    lower_endpoint_minimum = torch.min(frame_cutoff_times_torch[None, None, :-1], spike_bin_times[:, :-1, None])

    # shape (n_all_spikes, n_frames, n_sta_bins)
    distance_to_frame_bin_end = (frame_cutoff_times_torch[None, None, 1:] - spike_bin_times[:, :-1, None]) > 0.0
    distance_to_frame_bin_begin = (spike_bin_times[:, 1:, None] - frame_cutoff_times_torch[None, None, :-1]) > 0.0

    # shape (n_all_spikes, n_frames, n_sta_bins)
    does_overlap = distance_to_frame_bin_end & distance_to_frame_bin_begin

    # shape (n_all_spikes, n_sta_bins, n_frames)
    intersection_area = sum_area_torch[None, None, :] - (upper_endpoint_maximum - lower_endpoint_minimum)
    intersection_area[~does_overlap] = 0.0

    # shape (n_cells, n_all_spikes)
    reduction_matrix_torch = torch.tensor(reduction_matrix, dtype=torch.float32, device=device)

    # shape (n_cells, n_all_spikes) einsum (n_all_spikes, n_sta_bins, n_frames) -> (n_cells, n_sta_bins, n_frames)
    return torch.einsum('cs,sbf->cbf', reduction_matrix_torch, intersection_area)


def bin_relevant_spikes_and_advance_offsets(spikes_by_cell_id: Dict[int, np.ndarray],
                                            cell_idx_offset: Dict[int, int],
                                            cell_order: Sequence[int],
                                            earliest_relevant_spike_sample: int,
                                            last_relevant_spike_sample: int,
                                            advance_offset_to_time : float) \
        -> Tuple[Dict[int, int], Dict[int, np.ndarray], int]:
    '''
    Efficient method for grabbing the spike times T for each cell that lies within the window
        earliest_relevant_spike_sample < T < last_relevant_spike_sample

    Also computes the total number of spikes from any cell within the window, and also returns an updated
        index offset position Dict that accounts for the fact that a spike at the end of the relevant window
        may contribute frames from both this interval and the next interval to the final STA

    In more detail, the function performs the following functions:
        For every cell, for tuple (index i , spike_time T) grab the spike times that satisfies both of the inequalities

            earliest_relevant_spike_sample < T < last_relevant_spike_sample
            i >= cell_idx_offset[cell_id]

        Also return an updated offset dict, where cell_idx_offset_post[cell_id] corresponds to the index of the first
            spike for cell_id that occurs at or after advance_offset_to_time

    :param spikes_by_cell_id: Dict, (cell_id) -> (spike time vector)
    :param cell_idx_offset: Dict, (cell_id) -> (first not-yet-looked-at index in the spike time vector).
    :param cell_order: sequence of cell_id, corresponding to the order of cells for batched matrix multiply
    :param earliest_relevant_spike_sample: the earliest possible spike time is included, in units of electrical samples
    :param last_relevant_spike_sample: the last possible spike time that is included, in units of electrical samples
    :param advance_offset_to_time: the sample number to advance the index counters to, in units of electrical samples
    :return:
    '''
    cell_idx_offset_post = {}  # type: Dict[int, int]
    relevant_spikes_dict = {}  # type: Dict[int, np.ndarray]
    max_n_relevant_spikes = 0
    for cell_id in cell_order:

        spike_times_vector = spikes_by_cell_id[cell_id]
        last_looked_at = cell_idx_offset[cell_id]

        while last_looked_at < spike_times_vector.shape[0] and \
                spike_times_vector[last_looked_at] < earliest_relevant_spike_sample:
            last_looked_at += 1

        # now we are either in relevant spike territory, or we have no spikes for this bin
        relevant_spikes = []
        while last_looked_at < spike_times_vector.shape[0] and \
                spike_times_vector[last_looked_at] < last_relevant_spike_sample:
            relevant_spikes.append(spike_times_vector[last_looked_at])
            last_looked_at += 1

        relevant_spikes_dict[cell_id] = np.array(relevant_spikes, dtype=np.float32)
        max_n_relevant_spikes = max(max_n_relevant_spikes, relevant_spikes_dict[cell_id].shape[0])

        to_include = cell_idx_offset[cell_id]
        while to_include < spike_times_vector.shape[0] and \
                spike_times_vector[to_include] < advance_offset_to_time:
            to_include += 1

        cell_idx_offset_post[cell_id] = to_include

    return cell_idx_offset_post, relevant_spikes_dict, max_n_relevant_spikes


def torch_batch_parallel_pack_sta_bin_select_matrix(
        spikes_by_cell_id: Dict[int, np.ndarray],
        cell_idx_offset: Dict[int, int],
        cell_order: Sequence[int],
        n_bins_depth: int,
        bin_interval_samples: Union[int, float],
        frame_cutoff_times: np.ndarray,
        batch_size: int,
        device: torch.device) \
        -> Tuple[torch.Tensor, Dict[int, int]]:
    '''

    Batches over cells

    :param spikes_by_cell_id: Dict, (cell_id) -> (spike time vector)
    :param cell_idx_offset: Dict, (cell_id) -> (first not-yet-looked-at index in the spike time vector)
    :param cell_order: sequence of cell_id, corresponding to the order of cells for batched matrix multiply
    :param n_bins_depth: depth of STA, in bins
    :param bin_interval_samples: width of bins, units are recorded electrical samples
    :param frame_cutoff_times: cutoff times for the STA frames, including both endpoints. units are recorded
        electrical samples
    :param batch_size: batch size, int. This batches over the number of cells
    :return:
    '''

    n_cells = len(cell_order)
    n_frames = frame_cutoff_times.shape[0] - 1

    # not sure what the correct way to go to integer here is
    sta_length_in_electrical_samples = np.ceil(n_bins_depth * bin_interval_samples)

    earliest_relevant_spike_sample = frame_cutoff_times[0] - sta_length_in_electrical_samples
    last_relevant_spike_sample = sta_length_in_electrical_samples + frame_cutoff_times[-1]

    # bin every spike that could contribute frames to this interval,
    # and then advance the index counter for every cell
    cell_idx_offset_post, relevant_spikes_dict, max_n_relevant_spikes = bin_relevant_spikes_and_advance_offsets(
        spikes_by_cell_id,
        cell_idx_offset,
        cell_order,
        earliest_relevant_spike_sample,
        last_relevant_spike_sample,
        frame_cutoff_times[-1]
    )

    bin_select_matrix = torch.zeros((n_cells, n_bins_depth, n_frames), dtype=torch.float32, device=device)
    for ii in range(0, n_cells, batch_size):
        max_ii = min(n_cells, ii + batch_size)
        n_cells_in_batch = max_ii - ii

        spikes_cat = np.concatenate([relevant_spikes_dict[cell_id] for cell_id in cell_order[ii:max_ii]], axis=0)
        reduction_mat = np.zeros((n_cells_in_batch, spikes_cat.shape[0]), dtype=np.float32)

        offset = 0
        for idx, cell_id in enumerate(cell_order[ii:max_ii]):
            n_relevant_spikes_for_cell = relevant_spikes_dict[cell_id].shape[0]
            reduction_mat[idx, offset:offset + n_relevant_spikes_for_cell] = 1.0
            offset += n_relevant_spikes_for_cell

        if spikes_cat.shape[0] > 0:
            bin_select_matrix[ii:max_ii, ...] = torch_parallel_spike_bin_select_matrix_piece(
                spikes_cat,
                reduction_mat,
                n_bins_depth,
                bin_interval_samples,
                frame_cutoff_times,
                device
            )

    return bin_select_matrix, cell_idx_offset_post


def bin_frames_by_spike_times(spikes_by_cell_id: Dict[int, np.ndarray],
                              ttl_times: np.ndarray,
                              frame_generator: RandomNoiseFrameGenerator,
                              frames_per_ttl: int,
                              bin_interval_samples: Union[int, float],
                              n_bins_depth: int,
                              cell_batch_size: int,
                              device: torch.device) -> Dict[int, np.ndarray]:
    '''
    Calculates STAs by binning frames and spike times into bins where the
        master clock frequency is determined by the electrical sampling rate
        rather than by the stimulus / display

    :param spikes_by_cell_id: Dict, int cell id -> sorted spike time vector
    :param ttl_times: array of TTL times, units of samples
    :param frame_generator: frame generator object
    :param frames_per_ttl: Number of stimulus frames corresponding to one TTL interval,
        typically something like 100
    :param bin_interval_samples: number of samples per bin, can be noninteger
        since obviously the electrical ADC sample rate may not be an integer multiple
        of the target STA frame rate and the
    :param n_bins_depth: Depth (integer number of "frames") of the STA
    :return: Dict, int cell id -> STA matrix
    '''

    # initialize empty STAs
    height, width = frame_generator.output_dims
    cell_order = list(spikes_by_cell_id.keys())
    n_cells = len(cell_order)

    spikes_idx_offset = {cell_id: 0 for cell_id in spikes_by_cell_id.keys()}

    # outer loop is time
    # note that STAs are only about 25-50 frames worth
    # so we only need to keep two batches of frames around
    sta_buffer = torch.zeros((n_cells, n_bins_depth, height, width, 3), dtype=torch.float32, device=device)

    n_frame_blocks = ttl_times.shape[0] - 1
    n_distinct_frames = frames_per_ttl // frame_generator.refresh_interval

    pbar = tqdm.tqdm(total=n_frame_blocks)
    # compute STA contribution for block of frames corresponding to 1 TTL interval
    # Note that spikes from slightly after the end of the last frame in the interval
    # can contribute to the STA, since the STA includes a window of time before the spike
    for i in range(0, ttl_times.shape[0] - 1):
        ttl_a, ttl_b = ttl_times[i], ttl_times[i + 1]

        # shape (frames_per_ttl, width, height, 3)
        frame_batch = frame_generator.generate_block_of_frames(n_distinct_frames)

        frame_batch_torch = (torch.tensor(frame_batch, dtype=torch.float32, device=device) - 127.5) / 255.0
        ttl_bins = np.linspace(ttl_a, ttl_b, n_distinct_frames + 1)

        weights_matrix, spikes_idx_offset = torch_batch_parallel_pack_sta_bin_select_matrix(
            spikes_by_cell_id,
            spikes_idx_offset,
            cell_order,
            n_bins_depth,
            bin_interval_samples,
            ttl_bins,
            cell_batch_size,
            device
        )

        sta_buffer += torch.einsum('cdf,fwht->cdwht', weights_matrix, frame_batch_torch)

        pbar.update(1)
    pbar.close()

    sta_buffer_np = sta_buffer.cpu().numpy()

    sta_dict = {}  # type: Dict[int, np.ndarray]
    for idx, cell_id in enumerate(cell_order):
        sta_dict[cell_id] = sta_buffer_np[idx, ...] / spikes_by_cell_id[cell_id].shape[0]

    return sta_dict


def bin_spike_times_by_frames(spikes_by_cell_id: Dict[int, np.ndarray],
                              ttl_times: np.ndarray,
                              frame_generator: RandomNoiseFrameGenerator,
                              frames_per_ttl: int,
                              n_bins_depth: int,
                              cell_batch_size: int,
                              device: torch.device) -> Dict[int, np.ndarray]:
    '''
    Computes STAs by binning spikes into bins corresponding to the frame transition times

    Conceptually similar to the standard STA calculation method, but uses the overlap interval
        algorithm underneath

    :param spikes_by_cell_id:
    :param ttl_times:
    :param frame_generator:
    :param frames_per_ttl:
    :param bin_interval_samples: number of samples per bin
    :param n_bins_depth:
    :return:
    '''

    # initialize empty STAs
    height, width = frame_generator.output_dims
    cell_order = list(spikes_by_cell_id.keys())
    n_cells = len(cell_order)

    spikes_idx_offset = {cell_id: 0 for cell_id in spikes_by_cell_id.keys()}

    # outer loop is time
    # note that STAs are only about 25-50 frames worth
    # so we only need to keep two batches of frames around
    sta_buffer = torch.zeros((n_cells, n_bins_depth, height, width, 3), dtype=torch.float32, device=device)

    n_frame_blocks = ttl_times.shape[0] - 1
    n_distinct_frames = frames_per_ttl // frame_generator.refresh_interval

    with tqdm.tqdm(total=n_frame_blocks) as pbar:

        for i in range(0, ttl_times.shape[0] - 1):
            ttl_a, ttl_b = ttl_times[i], ttl_times[i + 1]

            frame_batch = frame_generator.generate_block_of_frames(
                n_distinct_frames)  # shape (frames_per_ttl, width, height, 3)

            frame_batch_torch = (torch.tensor(frame_batch, dtype=torch.float32, device=device) - 127.5) / 255.0
            ttl_bins = np.linspace(ttl_a, ttl_b, n_distinct_frames + 1)

            display_frame_interval = (ttl_b - ttl_a) / frames_per_ttl

            weights_matrix, spikes_idx_offset = torch_batch_parallel_pack_sta_bin_select_matrix(
                spikes_by_cell_id,
                spikes_idx_offset,
                cell_order,
                n_bins_depth,
                display_frame_interval,
                ttl_bins,
                cell_batch_size,
                device
            )

            sta_buffer += torch.einsum('cdf,fwht->cdwht', weights_matrix, frame_batch_torch)

            pbar.update(1)

    sta_buffer_np = sta_buffer.cpu().numpy()

    sta_dict = {}  # type: Dict[int, np.ndarray]
    for idx, cell_id in enumerate(cell_order):
        sta_dict[cell_id] = sta_buffer_np[idx, ...] / spikes_by_cell_id[cell_id].shape[0]

    return sta_dict
