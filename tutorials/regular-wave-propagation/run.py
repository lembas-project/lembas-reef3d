from lembas import CaseList

from lembas_reef3d import RegularWaveCase

case_list = CaseList()
case_list.add_cases_by_parameter_sweep(
    RegularWaveCase,
    num_processors=[4, 8],
    wave_height=[0.02, 0.03, 0.04],
    wave_length=[4.0, 6.0],
    plot=True,
)
case_list.run_all()
