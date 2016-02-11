from pyfann import libfann

nn = libfann.neural_net()
nn.create_from_file("and.net")

print nn.run([1, -1])
print nn.run([1, 1])
print nn.run([-1, 1])
print nn.run([-1, -1])
