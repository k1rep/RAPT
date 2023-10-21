import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
import torch.nn.functional as F


class AvgPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.pooler = torch.nn.AdaptiveAvgPool2d((1, config.hidden_size))

    def forward(self, hidden_states):
        return self.pooler(hidden_states).view(-1, self.hidden_size)


class RelationClassifyHeader(nn.Module):
    """
    H2:
    use averaging pooling across tokens to replace first_token_pooling
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)

        self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, code_hidden, text_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        diff_hidden = torch.abs(pool_code_hidden - pool_text_hidden)
        concated_hidden = torch.cat((pool_code_hidden, pool_text_hidden), 1)
        concated_hidden = torch.cat((concated_hidden, diff_hidden), 1)

        x = self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class TBertT(PreTrainedModel):
    def __init__(self, config, code_bert):
        super().__init__(config)
        # nbert_model = "huggingface/CodeBERTa-small-v1"
        cbert_model = code_bert

        self.ctokneizer = AutoTokenizer.from_pretrained(cbert_model,local_files_only = True)
        self.cbert = AutoModel.from_pretrained(cbert_model,local_files_only = True)

        self.ntokenizer = self.ctokneizer
        self.nbert = self.cbert

        # self.cls = RelationClassifyHeader(config)
        self.cls = CosineTrainHeader(config)

    def forward(
            self,
            text_ids=None,
            text_attention_mask=None,
            pos_code_ids=None,
            pos_code_attention_mask=None,
            neg_code_ids=None,
            neg_code_attention_mask=None):
        pos_hidden = self.cbert(pos_code_ids, attention_mask=pos_code_attention_mask)[0]
        neg_hidden = self.cbert(neg_code_ids, attention_mask=neg_code_attention_mask)[0]
        text_hidden = self.nbert(text_ids, attention_mask=text_attention_mask)[0]
        loss = self.cls(text_hidden, pos_hidden, neg_hidden)
        output_dict = {"loss": loss}
        return output_dict

    def get_sim_score(self, text_hidden, code_hidden):
        sim_scores = self.cls.get_sim(text_hidden=text_hidden, code_hidden=code_hidden).data
        return sim_scores

    def get_nl_tokenizer(self):
        return self.ntokenizer

    def get_pl_tokenizer(self):
        return self.ctokneizer

    def create_nl_embd(self, input_ids, attention_mask):
        return self.nbert(input_ids, attention_mask)

    def create_pl_embd(self, input_ids, attention_mask):
        return self.cbert(input_ids, attention_mask)

    def get_nl_sub_model(self):
        return self.nbert

    def get_pl_sub_model(self):
        return self.cbert


class CosineTrainHeader(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.code_pooler = AvgPooler(config)
        self.text_pooler = AvgPooler(config)
        self.margin = 0.5

    def forward(self, text_hidden, pos_code_hidden, neg_code_hidden):
        pool_pos_code_hidden = self.code_pooler(pos_code_hidden)
        pool_neg_code_hidden = self.code_pooler(neg_code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)

        anchor_sim = F.cosine_similarity(pool_text_hidden, pool_pos_code_hidden)
        neg_sim = F.cosine_similarity(pool_text_hidden, pool_neg_code_hidden)
        loss = (self.margin - anchor_sim + neg_sim).clamp(min=1e-6).mean()
        loss.requires_grad_(True)
        return loss

    def get_sim(self, text_hidden, code_hidden):
        pool_code_hidden = self.code_pooler(code_hidden)
        pool_text_hidden = self.text_pooler(text_hidden)
        sim = F.cosine_similarity(pool_text_hidden, pool_code_hidden)
        return torch.tensor(sim)



