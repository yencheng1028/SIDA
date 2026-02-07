因為記憶體容量有限，

1. precision的差異 (fp32, fp16, bf16)
電腦儲存一個浮點數（小數）所佔用的位元數。

fp32 (Float32): 全精度。每個數字佔 32 bits。最準確，但極度消耗記憶體。

fp16 (Half Precision): 半精度。每個數字僅佔 16 bits。記憶體消耗減半，速度變快，但如果數字太大或太小容易出錯（溢位）。

bf16 (Brain Float 16): 由 Google 開發。同樣是 16 bits，但它犧牲了一點點精度來換取與 fp32 相同的數值範圍，這讓訓練和推論大模型時更穩定，是目前主流顯卡（如 RTX 30/40 系列）的首選。

2. 量化技術 (4-bit / 8-bit)
原本需要 16 bits 儲存的權重，壓縮成僅需 8 bits 甚至 4 bits。

原本需要 20GB 顯存的模型，經過 4-bit 量化後可能只需要 5GB 就能跑動。

3. Inference指令: CUDA_VISIBLE_DEVICES=0,1 python chat_dual_gpu.py --version='./ck/SIDA-7B' --precision fp16