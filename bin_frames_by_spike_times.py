import numpy as np

import visionloader as vl
import visionwriter as vw
from whitenoise import RandomNoiseFrameGenerator

import pickle

import tqdm

import argparse
from collections import namedtuple

from typing import Dict, Tuple, Union, Sequence

FrameBatch = namedtuple('FrameBatch', ['frames', 'cutoff_times'])


def single_spike_bin_select_matrix_piece(spike_time: int,
                                         n_bins_depth: int,
                                         bin_interval_samples: Union[float, int],
                                         frame_cutoff_times: np.ndarray) -> np.ndarray:
    '''

    :param spike_time: time of spike, in units of electrical samples
    :param n_bins_depth: number of bins in the STA
    :param bin_interval_samples: size of bin, in units of electrical samples
    :param frame_cutoff_times: cutoff times of each frame, including endpoints. units of electrical samples
    :return: matrix of weights corresponding, shape (n_frames, n_bins_depth)
    '''
    frame_interval_length = frame_cutoff_times[1] - frame_cutoff_times[0]
    sum_area = frame_interval_length + bin_interval_samples

    spike_bin_times = spike_time - np.r_[n_bins_depth:-1:-1] * bin_interval_samples

    distance_to_frame_bin_end = frame_cutoff_times[1:, None] - spike_bin_times[None, :-1]
    distance_to_frame_bin_begin = spike_bin_times[None, 1:] - frame_cutoff_times[:-1, None]

    does_overlap = np.logical_and(distance_to_frame_bin_end > 0.0, distance_to_frame_bin_begin > 0.0)
    upper_endpoint_maximum = np.maximum(frame_cutoff_times[1:, None], spike_bin_times[None, 1:])
    lower_endpoint_minimum = np.minimum(frame_cutoff_times[:-1, None], spike_bin_times[None, :-1])

    intersection_area = sum_area - (upper_endpoint_maximum - lower_endpoint_minimum)
    intersection_area[np.logical_not(does_overlap)] = 0.0

    return intersection_area


def pack_sta_bin_select_matrix(spikes_by_cell_id: Dict[int, np.ndarray],
                               cell_idx_offset: Dict[int, int],
                               cell_order: Sequence[int],
                               n_bins_depth: int,
                               bin_interval_samples: Union[int, float],
                               frame_cutoff_times: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
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

    bin_select_matrix = np.zeros((n_cells, n_frames, n_bins_depth), dtype=np.float32)

    cell_idx_offset_post = {}  # type: Dict[int, int]
    for batch_idx, cell_id in enumerate(cell_order):

        spike_times_vector = spikes_by_cell_id[cell_id]
        last_looked_at = cell_idx_offset[cell_id]

        while last_looked_at < spike_times_vector.shape[0] and \
                spike_times_vector[last_looked_at] < earliest_relevant_spike_sample:
            last_looked_at += 1

        # now we are either in relevant spike territory, or we have no spikes for this bin
        while last_looked_at < spike_times_vector.shape[0] and \
                spike_times_vector[last_looked_at] < last_relevant_spike_sample:
            # now pack the matrix
            spike_time = spike_times_vector[last_looked_at]
            weights_mat_single_spike = single_spike_bin_select_matrix_piece(
                spike_time,
                n_bins_depth,
                bin_interval_samples,
                frame_cutoff_times
            )

            bin_select_matrix += weights_mat_single_spike
            last_looked_at += 1

        to_include = cell_idx_offset[cell_id]
        while to_include < spike_times_vector.shape[0] and \
                spike_times_vector[to_include] < frame_cutoff_times[-1]:
            to_include += 1

        cell_idx_offset_post[cell_id] = to_include

    return bin_select_matrix, cell_idx_offset_post


def bin_frames_by_spike_times(spikes_by_cell_id: Dict[int, np.ndarray],
                              ttl_times: np.ndarray,
                              frame_generator: RandomNoiseFrameGenerator,
                              frames_per_ttl: int,
                              bin_interval_samples: Union[int, float],
                              n_bins_depth: int) -> Dict[int, np.ndarray]:
    '''

    :param spikes_by_cell_id:
    :param ttl_times:
    :param frame_generator:
    :param frames_per_ttl:
    :param bin_interval_samples: number of samples per bin
    :param n_bins_depth:
    :return:
    '''

    # initialize empty STAs
    width, height = frame_generator.stixel_width, frame_generator.stixel_height
    cell_order = list(spikes_by_cell_id.keys())
    n_cells = len(cell_order)

    spikes_idx_offset = {cell_id: 0 for cell_id in spikes_by_cell_id.keys()}

    # outer loop is time
    # note that STAs are only about 25-50 frames worth
    # so we only need to keep two batches of frames around
    sta_buffer = np.zeros((n_cells, n_bins_depth, width, height, 3), dtype=np.float32)

    n_frame_blocks = ttl_times.shape[0] - 1

    with tqdm.tqdm(total=n_frame_blocks) as pbar:

        for i in range(0, ttl_times.shape[0] - 1):
            ttl_a, ttl_b = ttl_times[i], ttl_times[i + 1]

            frame_batch = frame_generator.generate_block_of_frames(
                frames_per_ttl)  # shape (frames_per_ttl, width, height, 3)
            ttl_bins = np.linspace(ttl_a, ttl_b, frames_per_ttl + 1)

            weights_matrix, spikes_idx_offset = pack_sta_bin_select_matrix(spikes_by_cell_id,
                                                                           spikes_idx_offset,
                                                                           cell_order,
                                                                           n_bins_depth,
                                                                           bin_interval_samples,
                                                                           ttl_bins)

            sta_buffer += np.einsum('cfd,fwht->hdwht', weights_matrix, frame_batch)

            pbar.update(1)

    sta_dict = {}  # type: Dict[int, np.ndarray]
    for idx, cell_id in enumerate(cell_order):
        sta_dict[cell_id] = sta_buffer[idx, ...] / spikes_by_cell_id[cell_id].shape[0]

    return sta_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute STAs at arbitrary frequency (frame rate and frame instabilities do not matter!')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('xml_path', type=str, help='path to stimulus XML file')

    args = parser.parse_args()

    dataset = vl.load_vision_data(args.ds_paht, args.ds_name, include_neurons=True)
    all_cells = dataset.get_cell_ids()
    spike_times_dict = {cell_id: dataset.get_spike_times_for_cell(cell_id) for cell_id in all_cells}
    ttl_times = dataset.get_ttl_times()

    framegen = RandomNoiseFrameGenerator.construct_from_xml(args.xml_path)

    sta_dict = bin_frames_by_spike_times(spike_times_dict,
                                         ttl_times,
                                         framegen,
                                         100,
                                         10.0e-3,
                                         61)

    pickle.dump('/Volumes/Lab/Users/ericwu/debug/sta_dict.p', sta_dict)
