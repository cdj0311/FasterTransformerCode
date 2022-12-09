import re
import time
import torch
from typing import List
import javalang
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from gpt import GPT
from trie_index import Trie


class FTGeneration(object):
    def __init__(self, 
                 layer_num=12,
                 head_num=16,
                 size_per_head=64,
                 vocab_size=50048,
                 lib_path="./libth_gpt.so",
                 ckpt_path='../convert/ft/models/megatron-models/c-model/250m/1-gpu',
                 weights_data_type='fp32',
                 data_type='fp32',
                 start_id=0, 
                 end_id=0,
                 max_seq_len=1024,
                 tensor_para_size=1, 
                 pipeline_para_size=1,
                 ):
        self.model = GPT(head_num, 
                       size_per_head, 
                       vocab_size, 
                       start_id, 
                       end_id, 
                       layer_num,
                       max_seq_len, 
                       tensor_para_size, 
                       pipeline_para_size, 
                       lib_path=lib_path,
                       weights_data_type=weights_data_type
            )
        self.model.load(ckpt_path=ckpt_path)
        if data_type == 'fp16':
            self.model.half()
        if data_type == 'bf16':
            self.model.bfloat16()
        self.end_id = end_id

    def ft_generate(self, start_ids: List[torch.Tensor], option_last_ids: List[torch.Tensor]=None,
                    output_len=32, beam_width=3, top_k=1, top_p=0, beam_search_diversity_rate=0,
                    temperature=1.0, len_penalty=0, repetition_penalty=1.0,
                    max_batch_size=1, return_output_length=True, return_cum_log_probs=1):
        
        random_seed_tensor = torch.zeros([max_batch_size], dtype=torch.int64)

        start_lengths = torch.IntTensor([len(ids) for ids in start_ids])
        start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)

        if option_last_ids is not None:
            option_last_counts = torch.IntTensor([len(ids) for ids in option_last_ids])
            option_last_ids = pad_sequence(option_last_ids, batch_first=True, padding_value=self.end_id)
        else:
            option_last_counts = None

        with torch.no_grad():
            tokens_batch = self.model(start_ids,
                                        start_lengths,
                                        output_len,
                                        beam_width,
                                        option_last_ids,
                                        option_last_counts,
                                        top_k * torch.ones(size=[max_batch_size], dtype=torch.int32),
                                        top_p * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        beam_search_diversity_rate * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        temperature * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        len_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        repetition_penalty * torch.ones(size=[max_batch_size], dtype=torch.float32),
                                        random_seed_tensor,
                                        return_output_length,
                                        return_cum_log_probs)
        tokens_batch, _, cum_log_probs = tokens_batch
        tokens_batch = tokens_batch.cpu().numpy().tolist()
        outputs = []
        for index, token_seqs in enumerate(tokens_batch):
            output = []
            for token_seq in token_seqs:
                _token = []
                for token in token_seq[start_lengths[index]:]:
                    if token == 0:
                        break
                    _token.append(token)
                output.append(_token)
            outputs.append(output)
        return outputs


class TextGeneration(object):
    def __init__(self, 
                 layer_num=12,
                 head_num=16,
                 size_per_head=64,
                 lib_path="./libth_gpt.so",
                 ft_ckpt_path='./250m/1-gpu',
                 tokenizer_path="./tokenizer.json",
                 weights_data_type='fp32',
                 data_type='fp32',
                 max_seq_len=1024):
        self.ft_model = FTGeneration(layer_num=layer_num,
                                    head_num=head_num,
                                    size_per_head=size_per_head,
                                    lib_path=lib_path,
                                    ckpt_path=ft_ckpt_path,
                                    weights_data_type=weights_data_type,
                                    data_type=data_type,
                                    max_seq_len=max_seq_len)
        self.tokenizer = Tokenizer.from_file(tokenizer_path) 
        self.max_seq_len = max_seq_len
        self.trie = Trie(self.tokenizer.get_vocab())

    def _option_tokens(self, tokens):
        input_text, last_text = " ".join(tokens[:-1]), tokens[-1]
        input_ids = self.tokenizer.encode(input_text).ids
        last_tokens = self.tokenizer.encode(last_text)
        option_tokens_ids = []
        self.trie.printAutoSuggestions("".join(last_tokens.tokens), option_tokens_ids)
        if not option_tokens_ids:
            input_ids += last_tokens.ids[:-1]
            last_text = "".join(last_tokens.tokens[-1:])
            self.trie.printAutoSuggestions(last_tokens.tokens[-1], option_tokens_ids)
            if not option_tokens_ids and last_tokens.tokens[-1] in self.tokenizer.get_vocab():
                option_tokens_ids.append((last_tokens.tokens[-1], self.tokenizer.get_vocab()[last_tokens.tokens[-1]]))
        option_tokens_ids = [i for t, i in option_tokens_ids]
        return input_ids, option_tokens_ids, last_text

    def generate(self, texts, output_len=32, beam_width=3):
        tokens = [[i.value for i in javalang.tokenizer.tokenize(text)] for text in texts] 
        is_partial = 1 if re.findall(r'[a-zA-Z]', texts[0][-1]) else 0
        input_ids, option_tokens_ids, last_texts = [], [], [] 
        if is_partial:
            for token in tokens:
                input_id, option_tokens_id, last_text = self._option_tokens(token)
                input_ids.append(torch.IntTensor(input_id))
                option_tokens_ids.append(torch.IntTensor(option_tokens_id))
                last_texts.append(last_text)
        else:
            for token in tokens:
                input_ids.append(torch.IntTensor(self.tokenizer.encode(" ".join(token)).ids))
            option_tokens_ids = None
            last_texts.append([])
        outputs = self.ft_model.ft_generate(input_ids, 
                                            option_tokens_ids, 
                                            output_len=output_len,
                                            beam_width=beam_width)

        for index, output_ids in enumerate(outputs):
            for output_id in output_ids:
                gen_text = self.tokenizer.decode(output_id, skip_special_tokens=False)
                print(gen_text)

        

if __name__ == '__main__':
    model = TextGeneration(layer_num=24,
                        head_num=32,
                        size_per_head=32,
                        lib_path="./libth_gpt.so",
                        ft_ckpt_path='./350m/1-gpu',
                        tokenizer_path="./tokenizer.json",
                        weights_data_type='fp32',
                        data_type='fp16',
                        max_seq_len=2048)
    texts = [
        "public class MyString { public void reverseString() { String s",
        "public class MyString { public void reverseString() { String",
    ]
    model.generate(texts, 32, 2)
    