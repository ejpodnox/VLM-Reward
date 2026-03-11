import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def main():
    parser = argparse.ArgumentParser(description="Run RoboReward-8B Inference on a Video")
    parser.add_argument("--video", type=str, required=True, help="Path to the robot rollout MP4 video")
    parser.add_argument("--task", type=str, required=True, help="Description of the task the robot is performing")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use flash_attention_2 for speed/memory savings if installed")
    args = parser.parse_args()

    print("Loading model and processor...")
    # Load model
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"
    }
    
    if args.use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
        
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "teetone/RoboReward-8B", 
        **kwargs
    )
    
    processor = AutoProcessor.from_pretrained("teetone/RoboReward-8B")

    # Define prompt based on the paper
    prompt_text = (
        "Given the task, assign a discrete progress score reward (1,2,3,4,5) "
        "for the robot in the video in the format: ANSWER: <score> "
        "Rubric for end-of-episode progress (judge only the final state without time limits): "
        "1 - No Success: Final state shows no goal-relevant change for the command. "
        "2 - Minimal Progress: Final state shows a small but insufficient change toward the goal. "
        "3 - Partial Completion: The final state shows good progress toward the goal but violates more than one requirement or a major requirement. "
        "4 - Near Completion: Final state is correct in region and intent but misses a single minor requirement. "
        "5 - Perfect Completion: Final state satisfies all requirements. "
        f"Task: {args.task}"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": args.video},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    print("Preparing inputs...")
    # Using qwen_vl_utils to process video properly for Qwen3-VL
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print("Generating response...")
    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    print("\n" + "="*50)
    print(f"Task: {args.task}")
    print(f"Video: {args.video}")
    print("="*50)
    print(f"Model output: {output_text[0]}")
    print("="*50)

if __name__ == "__main__":
    main()
