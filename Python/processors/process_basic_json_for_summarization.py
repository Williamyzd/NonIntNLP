import json
import torch
from transformers import XLNetTokenizer
import argparse

# Basic program which takes a input json file with a list of {'text','summary'} and outputs the stage-1 processed
# torch tensors needed by chunked_text_dataloader.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备初始数据")
    parser.add_argument(
    "--input_file",
    type=str,
    required=True,
    help="a input json file like [{ 'text': 'article text here','summary': 'article summary here' , { 'text': 'article2 text here','summary': 'article2 summary here'}"
    )
    parser.add_argument(
    "--model_name_or_path",
    type=str,
    required=True,
    help="the xlnet model name of transformers or a path of your local xlnet model."
    )
    args = parser.parse_args()
    input_path = args.input_file;
    model_name = args.model_name_or_path;

    with open(input_path) as f:
        datas = json.load(f)

    tok = XLNetTokenizer.from_pretrained(model_name)
    
    output = []
    for data in datas:
        text = data["text"]
        text_enc = tok.encode(
            text, add_special_tokens=False, max_length=None, pad_to_max_length=False
        )
        title = data["summary"]
        # Insert the title as the second sentence, forcing the proper token types.
        title_enc = tok.encode(
            title, add_special_tokens=False, max_length=None, pad_to_max_length=False
        )
        # Push resultants to a simple list and return it
        output.append({
            "text": torch.tensor(text_enc, dtype=torch.long),
            "target": torch.tensor(title_enc, dtype=torch.long),
        })

    torch.save(output, "out.pt")
