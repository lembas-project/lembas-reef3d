from lembas import Case, InputParameter, step


class Reef3dCase(Case):
    num_processors = InputParameter(type=int)

    @step
    def run(self):
        print(f"Running with {self.num_processors} processors")
