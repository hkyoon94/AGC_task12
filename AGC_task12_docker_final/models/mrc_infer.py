import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


##### QA 모델
class QAInference_1:
    def __init__(self, model_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    ### context, query를 받아 문답 수행
    def predict_answer(self, context, question):
        encodings = self.tokenizer(context, question,
                                   max_length=512,
                                   truncation=True,
                                   padding="max_length",
                                   return_token_type_ids=False,
                                   return_offsets_mapping=True)
        encodings = {
            key: torch.tensor([val]).to(self.device)
            for key, val in encodings.items()
        }

        pred = self.model(encodings["input_ids"], attention_mask=encodings["attention_mask"])
        start_logits, end_logits = pred.start_logits, pred.end_logits
        token_start_index, token_end_index = \
            start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
        start_confidence_score = torch.squeeze(
            torch.index_select(start_logits, -1, token_start_index)).item()
        end_confidence_score = torch.squeeze(
            torch.index_select(end_logits, -1, token_end_index)).item()
        answer_confidence_score = start_confidence_score + end_confidence_score

        pred_ids = encodings["input_ids"][0][token_start_index:token_end_index + 1]
        answer_text = self.tokenizer.decode(pred_ids)

        if not answer_text.strip():
            answer_confidence_score = -100

        answer_start_offset = int(encodings['offset_mapping'][0][token_start_index][0][0])
        answer_end_offset = int(encodings['offset_mapping'][0][token_end_index][0][1])
        answer_offset = (answer_start_offset, answer_end_offset)

        return {
            'answer_text': answer_text,  # QA 모델이 반환하는 정답
            'answer_offset': answer_offset,
            'answer_confidence_score': answer_confidence_score
        }  


##### QA 모델
class QAInference_2:
    def __init__(self, model_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def predict_answer(self, context, question):
        encoding = self.tokenizer(question,
                                  context,
                                  return_tensors="pt",
                                  max_length=256,
                                  padding="max_length",
                                  truncation=True)

        input_ids = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        with torch.no_grad():
            start_scores, end_scores = self.model(input_ids=input_ids,
                                                  attention_mask=attention_mask).to_tuple()

        # Let's take the most likely token using `argmax` and retrieve the answer
        all_tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())

        answer_tokens = all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1]
        answer = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(answer_tokens))

        return answer
