from lembas import CaseList

from lembas_reef3d import Reef3dCase

case_list = CaseList()
case_list.add_cases_by_parameter_sweep(
    Reef3dCase,
    wave_height=[0.02, 0.04],
    wave_length=4.0,
)
case_list.run_all()
