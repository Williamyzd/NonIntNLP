from transformers import XLNetTokenizer
import  os   #glob,
from multiprocessing import Pool
import random
import torch
import argparse
import json

# This function processes news articles gathered and preprocessed by the XSum data processor:
# https://github.com/EdinburghNLP/XSum
#
# The pre-processor generates a large number of files, each of which corresponds to a single article. The format of
# each file is text, with following format:
# [XSUM]URL[XSUM]
# <URL where article originates from>
# [XSUM]INTRODUCTION[XSUM]
# <Summary of the article>
# [XSUM]RESTBODY[XSUM]
# <Article text>

# JUNK_HEADER_TEXT = ["Share this with\n",
#                     "Email\n",
#                     "FaceBook\n",
#                     "Facebook\n",
#                     "Messenger\n",
#                     "Twitter\n",
#                     "Pinterest\n",
#                     "WhatsApp\n",
#                     "LinkedIn\n",
#                     "Linkedin\n",
#                     "Copy this link\n",
#                     "These are external links and will open in a new window\n"]

# Processes the contents of an XSUM file and returns a dict: {'text', 'summary'}
# def map_read_files(filepath):
#     with open(filepath, encoding="utf-8") as file:
#         content = file.read()
#         SUMMARY_INDEX = 4
#         TEXT_INDEX = 6
#         splitted = content.split("[XSUM]")

#         summary = splitted[SUMMARY_INDEX].strip()
#         text = splitted[TEXT_INDEX]
#         for junk in JUNK_HEADER_TEXT:
#             text = text.replace(junk, "").strip()

#         # Don't accept content with too small of text content or title content. Often these are very bad examples.
#         if len(text) < 1024:
#             return None
#         if len(summary) < 30:
#             return None

#         return {"summary": summary, "text": text}

# # This is a map function for processing reviews. It returns a dict:
# #  { 'text' { input_ids_as_tensor },
# #    'target' { input_ids_as_tensor } }
# tok = None
parser = argparse.ArgumentParser(
    description="Train an auto-regressive transformer model."
)
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
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="the dir to save datas"
)

args = parser.parse_args()
input_path = args.input_file;
model_name = args.model_name_or_path;
folder = args.data_dir;
tok = XLNetTokenizer.from_pretrained(model_name)
def map_tokenize_news(processed):
    text = processed["text"]
    text_enc = tok.encode(
        text, add_special_tokens=False, max_length=None, pad_to_max_length=False
    )

    title = processed["summary"]
    # Insert the title as the second sentence, forcing the proper token types.
    title_enc = tok.encode(
        title, add_special_tokens=False, max_length=None, pad_to_max_length=False
    )

    # Push resultants to a simple list and return it
    return {
        "text": torch.tensor(text_enc, dtype=torch.long),
        "target": torch.tensor(title_enc, dtype=torch.long),
    }


if __name__ == "__main__":

    os.chdir(folder)
    #files = glob.glob("*.data")
    output_folder = "/".join([folder, "outputs"])
    p = Pool(20)
    
    # Basic workflow:
    # MAP: [list of files to process] => list_of_news
    # MAP: [single list of shuffled news] => map_tokenize_news
    # REDUCE: [tokenized results] => [single list of tokenized results]
    print("Reading from files..")
    with open(input_path) as f:
        all_texts = json.load(f)
    all_texts = [m for m in all_texts if m is not None]

    print("Tokenizing news..")
    all_news = p.map(map_tokenize_news, all_texts)

    print("Writing news to output file.")
    random.shuffle(all_news)
    train_edge = int(len(all_news)*0.7)
    val_edge= train_edge + int(len(all_news)*0.2)

    train_news  = all_news[0:train_edge]
    val_news= all_news[train_edge:val_edge]
    test_news= all_news[val_edge:]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    torch.save(train_news, "/".join([output_folder, "train.pt"]))
    torch.save(val_news, "/".join([output_folder, "val.pt"]))
    torch.save(test_news, "/".join([output_folder, "test.pt"]))
