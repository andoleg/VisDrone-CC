import numpy as np
import torch
import time


# taken from https://github.com/sacmehta/ESPNet/issues/57
def computeTime(model, device='cuda', input_size=(128, 128)):
    inputs = torch.randn(1, 3, *input_size)
    if device == 1:
        model = model.cuda()
        inputs = inputs.cuda()

    model.eval()

    i = 0
    time_spent = []
    while i < 100:
        start_time = time.time()
        with torch.no_grad():
            _ = model(inputs)

        if device == 1:
            torch.cuda.synchronize()  # wait for cuda to finish (cuda is asynchronous!)
        if i != 0:
            time_spent.append(time.time() - start_time)
        i += 1
    average_time = np.mean(time_spent)
    fps = 1 / average_time
    # print('Avg execution time (ms): {:.3f}'.format(average_time))
    return average_time, fps
