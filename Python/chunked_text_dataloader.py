from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import transformers
import math
from typing import Tuple

# This class implements a Dataset which is capable of feeding a model which expects a stream of
# text with a generative target. It also (optionally) performs random masking on the target
# for the purposes of doing masked pre-training.
#
# It is unique because it supports extremely long sequences by "chunking" those sequences into several
# parts. Data retrieved from this set should then be fed into the model one chunk at a time until all chunks
# are exhausted, then more data should be fetched from the set.
#
# To facilitate this, this dataset sorts the data it returns by chunk size. It is strongly recommended that you
# do not randomize this data without batching first.
#
# Clients should use the get_dataloader() method to retrieve a dataloader from this class. This
# custom dataloader will guarantee that retrieved batches have the same internal chunk length.
#
# An example where this might be used is text summarization, where long text is fed in and
# the generative target is a summary of that text.
class ChunkedTextDataset(Dataset):
    # data_file=File path to pytorch pickle file with a list of dicts containing tokenized {"text", "target"}
    # tokenizer=huggingface-spec tokenizer used to tokenize data_file.
    # max_chunk_len=Sequence size per chunk. This minus `max_gen_len` is the space left for the actual text.
    # max_gen_len=A fixed upper cap for the sequence length of the generated text.
    # mask_percentage=The number of tokens to mask.
    # mask_all=Whether to mask tokens from the entire sequence, or only the generative portion of the sequence.
    def __init__(
        self,
        data_file: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_chunk_len=192,
        max_gen_len=64,
        mask_percentage=0.3,
        mask_all=False,
        pad_left=False,
    ):
        self.tokenizer = tokenizer
        self.max_chunk_len = max_chunk_len
        self.max_gen_len = max_gen_len
        self.mask_percentage = mask_percentage
        self.mask_all = mask_all
        self.pad_left = pad_left

        self.raw_data = torch.load(data_file)
        self.raw_data.sort(key=lambda x: x["text"].shape[0])

        # Each chunk will get a BOS, CLS and EOS token added to it.
        self.special_tokens_per_chunk = 3

    def process_element(self, text, target):
        # Tokens represented as 1-hot tensors which will be reused later in this function.
        bos_token_tensor = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        eos_token_tensor = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        sep_token_tensor = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long)
        zero_token_tensor = torch.tensor([0], dtype=torch.long)
        one_token_float_tensor = torch.tensor([1], dtype=torch.float)
        two_token_tensor = torch.tensor([2], dtype=torch.long)
        n100_token_tensor = torch.tensor([-100], dtype=torch.long)
        with torch.no_grad():
            target_len = target.shape[0]
            if target_len > self.max_gen_len:
                target = target[: self.max_gen_len]
                target_len = self.max_gen_len

            # Create attention_masks that'll go along with this tokenized text.
            attention_mask = torch.ones(text.shape[0], dtype=torch.float)

            # Each chunk will get a BOS, CLS and EOS token added to it.
            self.special_tokens_per_chunk = 3

            # We will chunk all inputs so that none exceed max_chunk_len, which will all be fed into the model
            # sequentially. Some set-up is necessary first.
            text_len_per_chunk = (
                self.max_chunk_len - target_len - self.special_tokens_per_chunk
            )
            num_chunks = math.ceil(text.shape[0] / text_len_per_chunk)
            final_text_seq_len = num_chunks * text_len_per_chunk

            # Before we can feed text into torch.chunk, it needs to be an exact multiple of text_len_per_chunk. This
            # will be accomplished by padding.
            padding_needed = final_text_seq_len - text.shape[0]
            padding_tensor = torch.full(
                (padding_needed,),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            att_padding_tensor = torch.zeros(padding_needed, dtype=torch.float)
            if self.pad_left:
                text = torch.cat([padding_tensor, text], dim=0)
                attention_mask = torch.cat([att_padding_tensor, attention_masks], dim=0)
            else:
                text = torch.cat([text, padding_tensor], dim=0)
                attention_mask = torch.cat([attention_mask, att_padding_tensor], dim=0)

            # The token_type_ids and labels are easy to init.
            token_type_ids = torch.zeros(final_text_seq_len, dtype=torch.long)
            labels = torch.full(
                (final_text_seq_len,), fill_value=-100, dtype=torch.long
            )

            chunked_text = torch.chunk(text, chunks=num_chunks)
            chunked_attention = torch.chunk(attention_mask, chunks=num_chunks)
            chunked_tokens = torch.chunk(token_type_ids, chunks=num_chunks)
            chunked_labels = torch.chunk(labels, chunks=num_chunks)

            # Now append the labels (and masks) per chunk
            input_ids = []
            input_ids_masked = []
            attention_mask = []
            token_type_ids = []
            labels = []
            for c_text, c_att, c_tok, c_lab in zip(
                chunked_text, chunked_attention, chunked_tokens, chunked_labels
            ):
                target_masked = target.clone().detach()
                label_append = torch.full((target_len,), fill_value=-100)

                # Perform masking on the target if needed.
                if not self.mask_all:
                    target_masked, label_append = self.perform_mask(target_masked)

                # This is where we're going to stick in all the special tokens, which makes these a little hard to
                # read. Remember: 3 special tokens. BOS at beginning, SEP to separate text and target. EOS at end.
                # Regardless - every output should have at least 3 values added to it.
                input_ids.append(
                    torch.cat(
                        [
                            bos_token_tensor,
                            c_text,
                            sep_token_tensor,
                            target,
                            eos_token_tensor,
                        ],
                        dim=0,
                    )
                )
                attention_mask.append(
                    torch.cat(
                        [
                            one_token_float_tensor,
                            c_att,
                            torch.ones(target_len + 2, dtype=torch.float),
                        ],
                        dim=0,
                    )
                )
                token_type_ids.append(
                    torch.cat(
                        [
                            zero_token_tensor,
                            c_tok,
                            two_token_tensor,
                            torch.ones(target_len + 1, dtype=torch.long),
                        ],
                        dim=0,
                    )
                )

                c_text_masked = torch.cat(
                    [
                        bos_token_tensor,
                        c_text,
                        sep_token_tensor,
                        target_masked,
                        eos_token_tensor,
                    ],
                    dim=0,
                )
                c_lab_full = torch.cat(
                    [
                        n100_token_tensor,
                        c_lab,
                        n100_token_tensor,
                        label_append,
                        n100_token_tensor,
                    ],
                    dim=0,
                )

                # Now we just have to perform full masking if necessary.
                if self.mask_all:
                    c_text_masked, c_lab_full = self.perform_mask(c_text_masked)
                input_ids_masked.append(c_text_masked)
                labels.append(c_lab_full)
            return {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_masks": attention_mask,
                "masked_input_ids": input_ids_masked,
                "labels": labels,
            }

    def perform_mask(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = tensor.clone()
        probability_matrix = torch.full(labels.shape, self.mask_percentage)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels, already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        tensor[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        tensor[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return tensor, labels

    def num_chunks_for_index(self, i):
        text = self.raw_data[i]["text"]
        target = self.raw_data[i]["target"]
        with torch.no_grad():
            target_len = target.shape[0]
            if target_len > self.max_gen_len:
                target_len = self.max_gen_len
            text_len_per_chunk = (
                self.max_chunk_len - target_len - self.special_tokens_per_chunk
            )
            return math.ceil(text.shape[0] / text_len_per_chunk)

    # The output of this Dataloader is a dict as follows:
    # 'input_ids':        A list of tokenized strings (chunks) with the target string append on the end after a <CLS> token.
    # 'token_type_ids':   A list of token_type_id encodings which can be fed into the model alongside input_ids.
    # 'attention_masks':  A list of attention_masks which can be fed into the model alongside input_ids.
    # For auto-regressive language modeling (e.g. pre-training):
    # 'masked_input_ids'  Same as 'input_ids', except parts are masked randomly.
    # 'labels':           A list of either (a) masked tokens or (b) -100 for auto-regressive LM loss calculation.
    def __getitem__(self, index):
        return self.process_element(
            self.raw_data[index]["text"], self.raw_data[index]["target"]
        )

    def __len__(self):
        return len(self.raw_data)

    def get_dataloader(self, batch_sz: int, num_workers=1):
        return DataLoader(
            self,
            batch_sampler=ChunkedTextBatchSampler(self, batch_sz),
            shuffle=False,
            num_workers=num_workers,
        )


# This sampler will only return batches with the same chunk size. It throws out extra elements. It relies on the
# underlying dataset to serve sorted text.
class ChunkedTextBatchSampler(Sampler):
    def __init__(
        self, chunked_text_set: ChunkedTextDataset, batch_sz: int, drop_last=True
    ):
        self.chunked_text_set = chunked_text_set
        self.batch_sz = batch_sz
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        batch_chunk_sz = 0
        for idx in range(len(self.chunked_text_set)):
            chunk_sz_idx = self.chunked_text_set.num_chunks_for_index(idx)
            if chunk_sz_idx != batch_chunk_sz:
                batch.clear()
                batch_chunk_sz = chunk_sz_idx
            batch.append(idx)
            if len(batch) == self.batch_sz:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.chunked_text_set) / self.batch_sz


if __name__ == "__main__":
    # Provided for testing.
    test_file = (
        "C:\\Users\\jbetk\\Documents\\data\\ml\\title_prediction\\outputs\\val.pt"
    )
    tokenizer = transformers.XLNetTokenizer.from_pretrained("xlnet-base-cased")
    dataset = ChunkedTextDataset(data_file=test_file, tokenizer=tokenizer)
    loader = dataset.get_dataloader(batch_sz=64)
    for batch in loader:
        print(tokenizer.decode(batch["masked_input_ids"][0][0]))