import argparse

import numpy as np

import visionloader as vl
import visionwriter as vw

from typing import List, Tuple, Dict

def fast_get_spike_count_multiple_cells(vision_dataset: vl.VisionCellDataTable,
                                        good_cell_ordering: List[List[int]],
                                        sample_interval_list_increasing: List[Tuple[int, int]]) -> np.ndarray:
    '''
    Function for binning spikes, if we include likely duplicates with nonoverlapping spike times

    We don't need a sophisticated algorithm for this, because we just care about the onset spikes
        with no sub-bin temporal resolution
    :param vision_dataset:
    :param good_cell_ordering:
    :param sample_interval_list_increasing:
    :return:
    '''
    n_images = len(sample_interval_list_increasing)
    n_cells = len(good_cell_ordering)

    # generate output matrix
    output_matrix = np.zeros((n_images, n_cells),
                             dtype=np.int32)

    # generate multiple cell cell_id to idx mapping
    # break the abstraction layer of vl.VisionCellDataTable for performance purposes...
    # keep grab all of the spike times for the cells that we care about
    # and keep a deep copy of that in a Dict

    # spike_idx_offset contains the index of the next spike whose time
    # we haven't looked at yet. This way we can count spikes while
    # only making a single pass through the data for each cell
    multicell_id_to_idx = {}  # type: Dict[int, int]
    spikes_by_cell_id = {}
    spike_idx_offset = {}
    for idx, cell_id_list in enumerate(good_cell_ordering):
        for cell_id in cell_id_list:
            multicell_id_to_idx[cell_id] = idx

            spike_times_ptr = vision_dataset.get_spike_times_for_cell(cell_id)
            spike_times_copy = np.copy(spike_times_ptr)

            spikes_by_cell_id[cell_id] = spike_times_copy
            spike_idx_offset[cell_id] = 0

    for interval_idx, interval in enumerate(sample_interval_list_increasing):

        for cell_id, map_to_idx in multicell_id_to_idx.items():

            spikes_for_current_cell = spikes_by_cell_id[cell_id]

            # advance the counter dict
            i = spike_idx_offset[cell_id]
            while i < len(spikes_for_current_cell) and spikes_for_current_cell[i] < interval[0]:
                i += 1

            # now we're at the first sample within the interval
            # count spikes within the interval
            n_spikes_in_interval = 0
            while i < len(spikes_for_current_cell) and spikes_for_current_cell[i] < interval[1]:
                n_spikes_in_interval += 1
                i += 1

            # and now we're outside the interval, do cleanup
            spike_idx_offset[cell_id] = i

            output_matrix[interval_idx, map_to_idx] += n_spikes_in_interval

    return output_matrix  # shape (n_intervals, n_cells)

if __name__ == '__main__':

    pass