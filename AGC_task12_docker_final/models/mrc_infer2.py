import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
)
from transformers import (
    SquadExample,)
from torch.utils.data import TensorDataset
from transformers.data.processors.squad import (
    MULTI_SEP_TOKENS_TOKENIZERS_SET,
    SquadFeatures,
    _new_check_is_max_context,
    TruncationStrategy,
    _improve_answer_span,
    whitespace_tokenize,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
)


class MRCModel:
    def __init__(self, model_name_or_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = BertConfig.from_pretrained(model_name_or_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path,
            do_lower_case=False,
        )
        self.mrc_model = BertForQuestionAnswering.from_pretrained(
            model_name_or_path,
            config=config,
        ).to(self.device)

    def preprocess(self, infer_inputs):
        examples = []
        for idx, sample in enumerate(infer_inputs):
            example = SquadExample(
                qas_id=idx,
                question_text=sample["question"],
                context_text=sample["context"],
                answer_text='',
                start_position_character=0,
                title='',
            )
            examples.append(example)
        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            # is_training=False,
            # return_dataset="pt",
        )
        return dataset, features, examples

    def post_processor(self, examples, features, results):
        return compute_predictions_logits(
            examples,
            features,
            results,
            5,
            30,
            False,
            None,
            None,
            None,
            False,
            False,
            0.0,
            self.tokenizer,
        )

    def model(self, eval_dataset, features):
        eval_dataloader = DataLoader(eval_dataset, batch_size=32)
        results = []
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                outputs = self.mrc_model(**inputs)
            example_indices = batch[3]
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs.to_tuple()]
                start_logits, end_logits = output
                token_start_index, token_end_index = np.argmax(
                    start_logits), np.argmax(end_logits)
                start_confidence_score = start_logits[token_start_index]
                end_confidence_score = end_logits[token_end_index]
                answer_confidence_score = start_confidence_score + end_confidence_score

                result = SquadResult(unique_id, start_logits, end_logits,
                                     answer_confidence_score)
                results.append(result)
        return results


class QAInference:

    def __init__(self, mrc: MRCModel):
        self.mrc = mrc

    def predict_answer(self, infer_inputs):
        """
        return {
            'answer_text': answer_text, # QA 모델이 반환하는 정답
            'answer_offset': answer_offset,
            'answer_confidence_score': answer_confidence_score
        }
        """
        infer_dataset, features, examples = self.mrc.preprocess(infer_inputs)
        model_outputs = self.mrc.model(infer_dataset, features)
        pred = self.mrc.post_processor(examples, features, model_outputs)
        mo = model_outputs[0]
        # for mo in model_outputs:
        answer_text = pred[0]
        answer_confidence_score = mo.confidence_score
        if not answer_text.strip():
            answer_confidence_score = -100
        return {
            "answer_text": answer_text,
            "answer_confidence_score": answer_confidence_score
        }


class MRCInference:

    def __init__(self, mrc: MRCModel):
        self.mrc = mrc

    def predict_answer(self, infer_inputs):
        # print(infer_inputs)
        infer_dataset, features, examples = self.mrc.preprocess(infer_inputs)
        model_outputs = self.mrc.model(infer_dataset, features)
        pred = self.mrc.post_processor(examples, features, model_outputs)
        return pred[0]


def to_list(tensor):
    return tensor.detach().cpu().tolist()


class SquadResult:

    def __init__(self,
                 unique_id,
                 start_logits,
                 end_logits,
                 confidence_score,
                 start_top_index=None,
                 end_top_index=None,
                 cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id
        self.confidence_score = confidence_score

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    padding_strategy="max_length",
):
    features = []
    for example in examples:
        features.append(
            squad_convert_example_to_features(example,
                                              max_seq_length=max_seq_length,
                                              doc_stride=doc_stride,
                                              max_query_length=max_query_length,
                                              padding_strategy=padding_strategy,
                                              is_training=False,
                                              tokenizer=tokenizer))

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in features:
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features],
                                 dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features],
                                       dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],
                                      dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features],
                                 dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_is_impossible = torch.tensor([f.is_impossible for f in features],
                                     dtype=torch.float)

    all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_masks,
                            all_token_type_ids, all_feature_index,
                            all_cls_index, all_p_mask)

    return features, dataset


def squad_convert_example_to_features(example, max_seq_length, doc_stride,
                                      max_query_length, padding_strategy,
                                      is_training, tokenizer):
    features = []
    if is_training and not example.is_impossible:
        # Get start and end position
        start_position = example.start_position
        end_position = example.end_position

        # If the answer cannot be found in the text, then skip this example.
        actual_text = " ".join(example.doc_tokens[start_position:(end_position +
                                                                  1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            print(
                f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'"
            )
            return []

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        if tokenizer.__class__.__name__ in [
                "RobertaTokenizer",
                "LongformerTokenizer",
                "BartTokenizer",
                "RobertaTokenizerFast",
                "LongformerTokenizerFast",
                "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.answer_text)

    spans = []

    truncated_query = tokenizer.encode(example.question_text,
                                       add_special_tokens=False,
                                       truncation=True,
                                       max_length=max_query_length)

    # Tokenizers who insert 2 SEP tokens in-between <context> & <question> need to have special handling
    # in the way they compute mask of added tokens.
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence +
        1 if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET else
        tokenizer.model_max_length - tokenizer.max_len_single_sentence)
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):

        # Define the side we want to truncate / pad and the text/pair sorting
        if tokenizer.padding_side == "right":
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # TODO(thom) update this logic
            texts,
            pairs,
            truncation=truncation,
            padding=padding_strategy,
            max_length=max_seq_length,
            return_overflowing_tokens=True,
            stride=max_seq_length - doc_stride - len(truncated_query) -
            sequence_pair_added_tokens,
            return_token_type_ids=True,
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            if tokenizer.padding_side == "right":
                non_padded_ids = encoded_dict[
                    "input_ids"][:encoded_dict["input_ids"].index(tokenizer.
                                                                  pad_token_id)]
            else:
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 -
                    encoded_dict["input_ids"][::-1].index(
                        tokenizer.pad_token_id))
                non_padded_ids = encoded_dict["input_ids"][
                    last_padding_id_position + 1:]

        else:
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

        token_to_orig_map = {}
        for i in range(paragraph_len):
            index = len(
                truncated_query
            ) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride
                                                         + i]

        encoded_dict["paragraph_len"] = paragraph_len
        encoded_dict["tokens"] = tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map
        encoded_dict["truncated_query_with_special_tokens_length"] = len(
            truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}
        encoded_dict["start"] = len(spans) * doc_stride
        encoded_dict["length"] = paragraph_len

        spans.append(encoded_dict)

        if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and
                len(encoded_dict["overflowing_tokens"]) == 0):
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]

    for doc_span_index in range(len(spans)):
        for j in range(spans[doc_span_index]["paragraph_len"]):
            is_max_context = _new_check_is_max_context(
                spans, doc_span_index, doc_span_index * doc_stride + j)
            index = (j if tokenizer.padding_side == "left" else
                     spans[doc_span_index]
                     ["truncated_query_with_special_tokens_length"] + j)
            spans[doc_span_index]["token_is_max_context"][
                index] = is_max_context

    for span in spans:
        # Identify the position of the CLS token
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implementation also keep the classification token (set to 0)
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens:] = 0
        else:
            p_mask[-len(span["tokens"]):-(len(truncated_query) +
                                          sequence_added_tokens)] = 0

        pad_token_indices = np.where(
            span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(
                span["input_ids"], already_has_special_tokens=True)).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # Set the cls index to 0: the CLS index can be used for impossible answers
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=
                0,  # Can not set unique_id and example_index here. They will be set after multiple processing.
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            ))
    return features