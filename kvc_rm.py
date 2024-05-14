import subprocess
import time
import torch
import threading

def get_gpu_status(gpu_index):
    # 获取指定GPU的内存信息
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader', f'--id={gpu_index}'], capture_output=True, text=True)
    # 解析输出结果
    gpu_data = result.stdout.strip()
    index, mem_total, mem_used = gpu_data.split(', ')
    mem_free = int(mem_total[:-3]) - int(mem_used[:-3])  # 计算剩余内存
    return (index, mem_free)

def occupy_memory(gpu_index, mem_free):
    # 计算需要生成的矩阵大小
    num_elements = (mem_free * 1024 * 1024) // 5  # 转换MB为字节数再计算元素个数
    # 在指定的GPU上创建随机矩阵
    device = torch.device(f"cuda:{gpu_index}")
    matrix = torch.randn(num_elements, dtype=torch.float32, device=device)
    print(f"Filled {mem_free} MB of GPU {gpu_index} with random matrix.")

def monitor_gpu(gpu_index):
    while True:
        index, mem_free = get_gpu_status(gpu_index)
        if mem_free > 20000:
            print(f"GPU {index} has more than 20GB free memory, occupying it now...")
            occupy_memory(gpu_index, mem_free)
        else:
            print(f"No GPU with more than 20GB free memory available right now on GPU {index}.")
        time.sleep(10)  # 每60秒检查一次

def main():
    num_gpus = torch.cuda.device_count()
    threads = []
    for gpu_index in range(num_gpus):
        gpu_index = 2
        thread = threading.Thread(target=monitor_gpu, args=(gpu_index,))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
