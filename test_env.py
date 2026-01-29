# 1.
import torch

print(f"CUDA 是否可用: {torch.cuda.is_available()}")

device_count = torch.cuda.device_count()
print(f"偵測到的 GPU 數量: {device_count}")

# 列出每張顯卡的名稱與細節
for i in range(device_count):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  - 顯存容量: {props.total_memory / 1024**2:.0f} MB")
    print(f"  - 計算能力: {props.major}.{props.minor}")



# # 2. 雙卡並行運算測試
# import torch
# import time

# # 測試矩陣乘法
# def test_gpu(device_id):
#     device = torch.device(f"cuda:{device_id}")
#     # 建立一個大型隨機矩陣 (10000x10000)
#     size = 10000
#     a = torch.randn(size, size, device=device)
#     b = torch.randn(size, size, device=device)
    
#     # 開始計時運算
#     start = time.time()
#     c = torch.matmul(a, b)
#     torch.cuda.synchronize(device) # 等待 GPU 完成運算
#     end = time.time()
    
#     print(f"GPU {device_id} 運算完成！耗時: {end - start:.4f} 秒")

# # 跑第一張卡
# test_gpu(0)
# # 跑第二張卡
# test_gpu(1)



# 3.測試 量化加速庫 多卡並行庫
# import bitsandbytes
# print("bitsandbytes 載入成功！")

# import deepspeed
# print("DeepSpeed 載入成功！")



# # 4.在另一個終端機動態監控，使用指令watch -n 1 nvidia-smi觀察
# # Processes是否出現 python 程序，
# # GPU-Util運算時使用率是否會衝高，
# # Memory-Usage: 顯存是否有被佔用，

# import torch
# import time

# def monitor_test():
#     size = 15000  # 矩陣大小
#     devices = [0, 1]
    
#     print("正在準備運算，請切換到 nvidia-smi 視窗...")
    
#     # 在兩張卡上都建立大矩陣
#     tensors_a = [torch.randn(size, size, device=f"cuda:{i}") for i in devices]
#     tensors_b = [torch.randn(size, size, device=f"cuda:{i}") for i in devices]

#     print("開始大量運算...")
#     for _ in range(20):  # 循環運算多次，拉長執行時間
#         for i in devices:
#             torch.matmul(tensors_a[i], tensors_b[i])
#         torch.cuda.synchronize() # 確保 GPU 運算中

#     print("運算結束，保持顯存佔用 10 秒，請觀察 Memory-Usage...")
#     time.sleep(10)
#     print("測試完成。")

# monitor_test()