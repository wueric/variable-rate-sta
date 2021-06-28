from lib.torch_sta import bin_spike_times_by_frames
from lib.save_data import generate_save_dict

import numpy as np
import torch

import visionloader as vl
import visionwriter as vw
from whitenoise import RandomNoiseFrameGenerator
import electrode_map as el_map

import pickle

import argparse
from typing import Dict, Tuple, Union, Sequence
import os

CELL_BATCH_SIZE = 64
N_DISPLAY_FRAMES_PER_TTL = 100
SAMPLE_FREQ = 20000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='compute STAs at monitor frequency')

    parser.add_argument('ds_path', type=str, help='path to Vision dataset')
    parser.add_argument('ds_name', type=str, help='name of Vision dataset')
    parser.add_argument('xml_path', type=str, help='path to stimulus XML file')
    parser.add_argument('output', type=str, help='path to save location')
    parser.add_argument('-n', '--n_frames', type=int, help='number of frames', default=51)
    parser.add_argument('-b', '--batch', type=int, help='number of cells batch size', default=CELL_BATCH_SIZE)
    parser.add_argument('-v', '--visionwriter', type=bool, default=False, help='save in Vision .sta format')
    parser.add_argument('-j', '--jitter', action='store_true', default=False, help='Use jittered stimulus')
    parser.add_argument('-o', '--manual_frame_offset', type=int, default=0,
                        help='Frame offset. Example: if N, the first trigger in the .neurons file is associated with N * N_DISPLAY_FRAMES_PER_TTL frames after the start of the stimulus')
    parser.add_argument('-t', '--manual_trigger_offset', type=int, default=0, help='Skip this many triggers')


    args = parser.parse_args()

    device = torch.device('cuda')
    torch.set_num_threads(8)

    print("Loading spike times...")
    dataset = vl.load_vision_data(args.ds_path, args.ds_name, include_neurons=True)
    all_cells = dataset.get_cell_ids()
    spike_times_dict = {cell_id: dataset.get_spike_times_for_cell(cell_id) for cell_id in all_cells}
    ttl_times = dataset.get_ttl_times()

    if args.manual_trigger_offset != 0:
        ttl_times = ttl_times[args.manual_trigger_offset:]

    avg_ttl_time = np.median(ttl_times[1:] - ttl_times[:-1])
    monitor_freq = 1.0 / (N_DISPLAY_FRAMES_PER_TTL * (avg_ttl_time / SAMPLE_FREQ))

    framegen = RandomNoiseFrameGenerator.construct_from_xml(args.xml_path, args.jitter)
    if args.manual_frame_offset != 0:
        framegen.advance_seed_n_frames(args.manual_frame_offset * N_DISPLAY_FRAMES_PER_TTL)

    print("Calculating STAs")
    sta_dict = bin_spike_times_by_frames(spike_times_dict,
                                         ttl_times,
                                         framegen,
                                         N_DISPLAY_FRAMES_PER_TTL,
                                         args.n_frames,
                                         args.batch,
                                         device)

    if args.visionwriter:

        print("Saving output to Vision format")

        sta_container_by_cell_id = {}  # type: Dict[int, vl.STAContainer]
        for cell_id, sta_matrix in sta_dict.items():
            depth, width, height, n_channels = sta_matrix.shape
            no_error = np.zeros_like(sta_matrix[..., 0])
            sta_container = vl.STAContainer(framegen.stixel_width,
                                            monitor_freq,
                                            0,
                                            sta_matrix[..., 0],
                                            no_error,
                                            sta_matrix[..., 1],
                                            no_error,
                                            sta_matrix[..., 2],
                                            no_error)
            sta_container_by_cell_id[cell_id] = sta_container

        print("Writing .sta file")
        with vw.STAWriter(args.output,
                          args.ds_name,
                          framegen.field_width,
                          framegen.field_height,
                          args.n_frames,
                          monitor_freq,
                          0,
                          framegen.stixel_width) as staw:

            staw.write_sta_by_cell_id(sta_container_by_cell_id)

        print("Rewriting .globals file")
        with vl.GlobalsFileReader(args.output, args.ds_name) as gfr:
            vision_header = gfr.get_rdh512_header() # type: vl.PyBinHeader
            if el_map.is_reconfigurable_board(vision_header.array_id):
                electrode_coordinates, _ = gfr.get_electrode_map()
            else:
                electrode_coordinates = None

        rtmp = vl.RunTimeMovieParamsReader(framegen.stixel_width,
                                           framegen.stixel_height,
                                           framegen.field_width,
                                           framegen.field_height,
                                           1.0,
                                           1.0,
                                           0.0,
                                           0.0,
                                           framegen.refresh_interval,
                                           monitor_freq,
                                           N_DISPLAY_FRAMES_PER_TTL,
                                           framegen.refresh_interval / monitor_freq,
                                           (ttl_times.shape[0] - 1) * N_DISPLAY_FRAMES_PER_TTL,
                                           [])

        with vw.GlobalsFileWriter(args.output, args.ds_name) as gfw:
            if el_map.is_reconfigurable_board(vision_header.array_id):
                gfw.write_simplified_reconfigurable_array_globals_file(vision_header.time_base,
                                                                       vision_header.seconds_time,
                                                                       vision_header.comment,
                                                                       vision_header.dataset_identifier,
                                                                       vision_header.format,
                                                                       vision_header.frequency,
                                                                       vision_header.n_samples,
                                                                       electrode_coordinates,
                                                                       17.5)
                gfw.write_run_time_movie_params(rtmp)

            else:
                gfw.write_simplified_litke_array_globals_file(vision_header.array_id,
                                                              vision_header.time_base,
                                                              vision_header.seconds_time,
                                                              vision_header.comment,
                                                              vision_header.dataset_identifier,
                                                              vision_header.format,
                                                              vision_header.n_samples)
                gfw.write_run_time_movie_params(rtmp)



    else:
        with open(args.output, 'wb') as pfile:
            save_dict = generate_save_dict(sta_dict, monitor_freq)
            pickle.dump(save_dict, pfile)
