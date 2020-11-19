from typing import Dict
import numpy as np


def generate_save_dict(sta_by_cell_id: Dict[int, np.ndarray],
                       sta_calculated_frame_rate: float):
    ret_dict = {
        'sta_by_cell_id': sta_by_cell_id,
        'calculated_frame_rate': sta_calculated_frame_rate
    }

    return ret_dict
