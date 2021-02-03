import dynet_config
# set random seed to have the same result each time
dynet_config.set(random_seed=169)
import dynet as dy
import numpy as np
import time

start = time.time()
# This implementation will be unnecessarily slow, especially on the GPU.
# It can be improved by following the speed tricks covered in class:
# 1) Don't repeat operations.
# 2) Minimize the number of operations.
# 3) Minimize the number of CPU-GPU memory copies, make them earlier.

# Create the model
model = dy.ParameterCollection()
trainer = dy.SimpleSGDTrainer(model)
W = model.add_parameters((100,100))

# Create the "training data"
x_vecs = np.random.rand(10, 100)
y_vecs = np.random.rand(10, 100)

sum = 0
# Do the processing
for my_iter in range(1000):
  dy.renew_cg()
  total = 0
  x_W_exp = dy.inputTensor(x_vecs) * W # 10, 100
  total = dy.sum_elems(x_W_exp * dy.inputTensor(y_vecs.transpose())) # 10, 10 -> 1
  sum += total.scalar_value()
  total.forward()
  total.backward()
  trainer.update()

total_time = time.time() - start
print(f"Time Taken: {total_time:.2f}s, Total: {sum:.2f}")