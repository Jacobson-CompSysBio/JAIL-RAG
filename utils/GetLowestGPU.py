import pynvml

def GetLowestGPU(
    devices: list = None, 
    verbose: bool = False,
):
    
    # initialize NVML
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    min_memory_usage = None
    min_memory_gpu_index = None

    # iterate over available GPUs
    for i in range(device_count):

        # skip if device not in list of devices
        if devices is not None and i not in devices:
            continue

        # extract information
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_usage = mem_info.used
        total_memory = mem_info.total

        # print GPU memory usage
        if verbose:
            memory_usage_mb = memory_usage / (1024 ** 2)
            total_memory_mb = total_memory / (1024 ** 2)
            print('cuda:{} {:5.0f}MiB / {:5.0f}MiB'.format(
                i, memory_usage_mb, total_memory_mb))

        # update GPU with minimum memory usage
        if min_memory_usage is None or memory_usage < min_memory_usage:
            min_memory_usage = memory_usage
            min_memory_gpu_index = i

    # clean up NVML
    pynvml.nvmlShutdown()

    # set the device string
    if min_memory_gpu_index is not None:
        device = f'cuda:{min_memory_gpu_index}'
    else:
        device = 'cpu'

    # print selection
    if verbose:
        if min_memory_gpu_index is not None:
            print()
        print(f'Device set to: {device}')

    return device

if __name__ == '__main__':
    import time
    t0 = time.time()
    GetLowestGPU(True)
    print('Elapsed time: {:.2f} ms'.format((time.time() - t0) * 1000))