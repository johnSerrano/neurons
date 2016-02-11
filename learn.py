from pyfann import libfann

learning_rate = 0.7
num_input = 2
num_hidden = 4
num_output = 1

desired_error = 0.000001
max_iterations = 10000
iterations_between_reports = 1000

nn = libfann.neural_net()
nn.create_standard_array((num_input, num_hidden, num_output))
nn.set_learning_rate(learning_rate)
nn.set_activation_function_output(libfann.SIGMOID_SYMMETRIC_STEPWISE)

nn.train_on_file("and.data", max_iterations, iterations_between_reports, desired_error)

nn.save("and.net")
