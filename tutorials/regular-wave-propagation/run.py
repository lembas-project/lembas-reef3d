from lembas import CaseList

from lembas_reef3d import RegularWaveCase

case_list = CaseList()
case_list.add_cases_by_parameter_sweep(
    RegularWaveCase,
    num_processors=[4],
    wave_height=[0.02],
    wave_length=4.0,
)
case_list.run_all()
