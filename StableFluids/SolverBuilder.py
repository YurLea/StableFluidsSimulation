class SolverBuilder:
    def __init__(self):
        self.parameter_0 = None

    def with_parameter_0(self, parameter_0):
        self.parameter_0 = parameter_0
        return self

print(SolverBuilder().with_parameter_0(1).parameter_0)