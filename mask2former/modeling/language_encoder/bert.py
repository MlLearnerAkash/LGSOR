# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Sequence

import torch
# from mmengine.model import BaseModel
from torch import nn

try:
    from transformers import AutoTokenizer, BertConfig
    from transformers import BertModel as HFBertModel
except ImportError:
    AutoTokenizer = None
    HFBertModel = None

class BertModel(nn.Module):
    """BERT model for language embedding only encoder.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
             Defaults to bert-base-uncased.
        max_tokens (int): maximum number of tokens to be used for BERT.
             Defaults to 256.
        pad_to_max (bool): whether to pad the tokens to max_tokens.
             Defaults to True.
        num_layers_of_embedded (int): number of layers of the embedded model.
             Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
             Defaults to False.
    """

    def __init__(self,
                 name: str = 'bert-base-uncased',
                 max_tokens: int = 256,
                 pad_to_max: bool = True,
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.pad_to_max = pad_to_max

        if AutoTokenizer is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')

        print(f'loading bert tokenider {name} ....')
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.language_backbone = nn.Sequential(
            OrderedDict([('body',
                          BertEncoder(
                              name,
                              num_layers_of_embedded=num_layers_of_embedded,
                              use_checkpoint=use_checkpoint))]))

    def forward(self, captions: Sequence[str], **kwargs) -> dict:
        """Forward function."""
        device = next(self.language_backbone.parameters()).device
        tokenized = self.tokenizer.batch_encode_plus(
            captions,
            max_length=self.max_tokens,
            padding='max_length' if self.pad_to_max else 'longest',
            return_special_tokens_mask=True,
            return_tensors='pt',
            truncation=True).to(device)

        tokenizer_input = {
            'input_ids': tokenized.input_ids,
            'attention_mask': tokenized.attention_mask
        }
        language_dict_features = self.language_backbone(tokenizer_input)
        return language_dict_features


class BertEncoder(nn.Module):
    """BERT encoder for language embedding.

    Args:
        name (str): name of the pretrained BERT model from HuggingFace.
                Defaults to bert-base-uncased.
        num_layers_of_embedded (int): number of layers of the embedded model.
                Defaults to 1.
        use_checkpoint (bool): whether to use gradient checkpointing.
                Defaults to False.
    """

    def __init__(self,
                 name: str = 'bert-base-uncased',
                 num_layers_of_embedded: int = 1,
                 use_checkpoint: bool = True,
                 add_pooling_layer: bool = False):
        super().__init__()
        if BertConfig is None:
            raise RuntimeError(
                'transformers is not installed, please install it by: '
                'pip install transformers.')
        print(f'loading bert model {name} ....')
        config = BertConfig.from_pretrained(name)
        config.gradient_checkpointing = use_checkpoint
        # only encoder
        self.model = HFBertModel.from_pretrained(
            name, add_pooling_layer=add_pooling_layer, config=config)
        self.language_dim = config.hidden_size
        self.num_layers_of_embedded = num_layers_of_embedded
        self.add_pooling_layer = add_pooling_layer

    def forward(self, x) -> dict:
        mask = x['attention_mask']

        outputs = self.model(
            input_ids=x['input_ids'],
            attention_mask=mask,
            output_hidden_states=True,
        )

        # outputs has 13 layers, 1 input layer and 12 hidden layers
        encoded_layers = outputs.hidden_states[1:]
        features = torch.stack(encoded_layers[-self.num_layers_of_embedded:],
                               1).mean(1)
        # language embedding has shape [len(phrase), seq_len, language_dim]
        features = features / self.num_layers_of_embedded
        embedded = features * mask.unsqueeze(-1).float()

        results = {
            'embedded': embedded,
            'masks': mask,
            'last_hidden_state': encoded_layers[-1],
            # 'pooler_output': outputs.pooler_output
        }
        if self.add_pooling_layer:
            results['pooler_output'] = outputs.pooler_output
        return results


if __name__ == '__main__':
    caps = ['hello world', 'hello world']
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    max_token_num = 256 
    tokens = tokenizer(
        caps, padding='max_length', truncation=True, max_length=max_token_num, return_tensors='pt'
    )
    ttt = {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    print(ttt['input_ids'].shape)
    print(ttt['attention_mask'].shape)

    model = BertEncoder('bert-base-uncased', add_pooling_layer=True)
    model.eval()
    
    with torch.no_grad():
        output = model(ttt)
        print(output['embedded'].shape)
        print(output['masks'].shape)
        print(output['last_hidden_state'].shape)
        print(output['pooler_output'].shape)
        # print(output['position_ids'].shape)
        # print(output['text_token_mask'].shape)
