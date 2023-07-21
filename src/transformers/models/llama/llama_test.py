import torch
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer

if __name__ == '__main__':
    config = LlamaConfig(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        is_decoder=False,
        initializer_range=0.02,
    )
    model = LlamaModel(config)

    input_ids = torch.zeros(2, 256, 768)
    attention_mask = torch.ones(2, 256)
    use_cache = True

    o = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
    )
    print(o)
