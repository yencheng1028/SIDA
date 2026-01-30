import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

# 移除 BitsAndBytesConfig，因為我們改用純 FP16
from model.SIDA import SIDAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="SIDA chat")
    parser.add_argument("--version", default="SIDA-7B-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    # 預設改為 fp16
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    # 這些參數保留但預設不啟用
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

    # 2. 設定 Dtype (強烈建議 1080Ti 使用 fp16)
    torch_dtype = torch.float16
    if args.precision == "bf16":
        # Pascal 架構不支援 bf16，會報錯或變慢，這裡強制轉回 fp16 或 fp32
        print("Warning: GTX 10-series does not support bfloat16. Switching to float16.")
        torch_dtype = torch.float16
    elif args.precision == "fp32":
        torch_dtype = torch.float32

    # 3. Model Loading (關鍵修改：device_map="auto")
    print(f"Loading model with {torch_dtype} across available GPUs...")
    
    kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",  # <--- 讓 HuggingFace 自動分配模型到兩張卡
    }

    model = SIDAForCausalLM.from_pretrained(
        args.version, 
        low_cpu_mem_usage=True, 
        vision_tower=args.vision_tower, 
        seg_token_idx=args.seg_token_idx, 
        cls_token_idx=args.cls_token_idx, 
        **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 注意：使用了 device_map="auto" 後，不要再呼叫 model.cuda()，否則會破壞分配
    
    try:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        # 確保 Vision Tower 也在正確的 dtype，device 交給 accelerate 管理
        vision_tower.to(dtype=torch_dtype) 
    except AttributeError:
        print("Vision tower initialization skipped.")

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    
    # 打印模型分配情況，確認是否有用到兩張卡
    print("\nModel device map:")
    if hasattr(model, "hf_device_map"):
        print(model.hf_device_map)
    else:
        print(f"Model is on: {model.device}")

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        try:
            prompt = input("Please input your prompt: ")
        except EOFError:
            break
            
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        # 處理 Image Clip
        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .to(device="cuda", dtype=torch_dtype) # 統一轉為 fp16 且放上 GPU
        )

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        # 處理 Image
        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .to(device="cuda", dtype=torch_dtype) # 統一轉為 fp16 且放上 GPU
        )

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda() # Input ID 通常在第一張卡處理即可

        with torch.no_grad():
            output_ids, pred_masks = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )
        
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.detach().cpu().numpy()[0]
            pred_mask = pred_mask > 0

            save_path = "{}/{}_mask_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            cv2.imwrite(save_path, pred_mask * 100)
            print("{} has been saved.".format(save_path))

            save_path = "{}/{}_masked_img_{}.jpg".format(
                args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
            )
            save_img = image_np.copy()
            save_img[pred_mask] = (
                image_np * 0.5
                + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
            )[pred_mask]
            save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, save_img)
            print("{} has been saved.".format(save_path))


if __name__ == "__main__":
    main(sys.argv[1:])