import numpy as np
from sentence_transformers import util
import re
import copy
from typing import (Tuple)
from .unit_form import DIGIT_UNITS, KIND_CHAR, RG_DIGIT, RG_AFTER_DIGIT
from inference.task2_form_mapping import map_key_value_from_context, extract_digit, map_list_from_context

MULTI_FILTER = 'multi_filter'
HIGH_COS = 'high_cos'


class ContextExtractor:

    @staticmethod
    def similarity_with_corpus_mrc_search(mrc_model, context, query, corpus):
        if isinstance(context, dict):
            raise ValueError("Wrong context type, context should be list")

        inputs = {"context": ". ".join(context), "question": query}
        querysimilar_answer = mrc_model.predict_answer([inputs])

        answer_sent = ''
        search_ans = copy.deepcopy(querysimilar_answer)
        if len(querysimilar_answer.split('. ')) > 1:
            search_ans = max(querysimilar_answer.split('. '),
                             key=lambda x: len(x))
        for sent in context:
            if search_ans.strip() in sent:
                answer_sent = sent
                break

        sent_idx = list(corpus).index(answer_sent)
        new_context_str, new_context_end = max(0, sent_idx - 3), min(
            len(corpus) - 1, sent_idx + 3)
        # mrc 결과를 바탕으로 나온 sentence 주변애들을 추출해온다 .
        new_context = list(corpus)[new_context_str:new_context_end]
        return new_context

    @staticmethod
    def get_context_from_task1(context):
        if isinstance(context, dict):
            return context['0']
        else:
            return context


def solve_task2_B(self, context, query, answer_form, corpus, context_type,
                  **kwargs):
    print('##############################################')
    print(query)

    if context_type == HIGH_COS:
        new_context = ContextExtractor.similarity_with_corpus_mrc_search(
            self.mrc_model, context, query, corpus)
    elif context_type == MULTI_FILTER:
        new_context = ContextExtractor.get_context_from_task1(context)
    else:
        new_context = ContextExtractor.get_context_from_task1(context)

    print('\n'.join(new_context))
    # get answer candidates from mrc model
    answers = repeat_infer_mrc(self.mrc_model, query, new_context, answer_form)
    # get filled answer_form
    if len(answer_form[0].values()) == 1:
        answer_form = map_list_from_context(answers, answer_form, new_context)
    else:
        answer_form = map_key_value_from_context(answers, answer_form, new_context)
    final_answer_form = copy.deepcopy(answer_form)
    for i, dict_ in enumerate(answer_form):
        hint = []
        for k, v in dict_.items():
            if v.strip() == '':
                inputs = {
                    "context": ' '.join(new_context),
                    "question": query + ' '.join(hint)
                }
                querysimilar_answer = self.mrc_model.predict_answer([inputs])
                digit = extract_digit(querysimilar_answer)
                if digit:
                    ans = digit
                else:
                    ans = querysimilar_answer.strip()
                final_answer_form[i][k] = ans
            else:
                hint.append(v)
    self.task2_log.append({
        'context': new_context,
        'query': query,
        'pred': answers,
        'answer_form': final_answer_form
    })
    print(final_answer_form)
    return final_answer_form


def solve_task2_A(self, context, query, answer_form, corpus, context_type,
                  **kwargs):
    """
    13점 받은 방법
    """
    print('##############################################')
    print(query)

    if context_type == HIGH_COS:
        new_context = ContextExtractor.similarity_with_corpus_mrc_search(
            self.mrc_model, context, query, corpus)
    elif context_type == MULTI_FILTER:
        new_context = ContextExtractor.get_context_from_task1(context)
    else:
        new_context = ContextExtractor.get_context_from_task1(context)

    print('\n'.join(new_context))
    # get answer candidates from mrc model
    answers = repeat_infer_mrc(self.mrc_model, query, new_context, answer_form)
    # get filled answer_form
    if len(answer_form[0].values()) == 1:
        for i, dict_ in enumerate(answer_form):
            for k, v in dict_.items():
                if v.strip() == '':
                    answer_form[i][k] = answers[i]
    else:
        answer_form = map_key_value_from_context(answers, answer_form,
                                                 new_context)
    print(answer_form)
    final_answer_form = copy.deepcopy(answer_form)
    for i, dict_ in enumerate(answer_form):
        hint = []
        for k, v in dict_.items():
            if v.strip() == '':
                inputs = {
                    "context": ' '.join(new_context),
                    "question": query + ' '.join(hint)
                }
                querysimilar_answer = self.mrc_model.predict_answer([inputs])
                digit = extract_digit(querysimilar_answer)
                if digit:
                    ans = digit
                else:
                    ans = querysimilar_answer.strip()
                final_answer_form[i][k] = ans
            else:
                hint.append(v)
    self.task2_log.append({
        'context': new_context,
        'query': query,
        'pred': answers,
        'answer_form': final_answer_form
    })
    print(final_answer_form)
    return final_answer_form


def repeat_infer_mrc(mrc_model, query, context, answer_form):
    answers = []
    tmp_context = " ".join(context)
    for i in range(
            len(answer_form)):  # mrc나온 값의  길이만큼 mrc 돌린다. 단 한번 나온 답은 제거한 후
        inputs = {"context": tmp_context, "question": query}
        querysimilar_answer = mrc_model.predict_answer([inputs])
        # print('pred : ', querysimilar_answer)
        answers.append(querysimilar_answer)
        tmp_context = tmp_context.replace(querysimilar_answer, '')

    return answers


def remove_digit(string):
    start, end = -1, -1
    for j, c in enumerate(string):
        if c.isdigit():
            start = j
            break
    for j, c in enumerate(string[::-1]):
        if c.isdigit():
            end = j
            break
    # ans = ''.join([char for char in digit_match.group() if char])
    return string[start:-end]


def get_task2type_and_formvalue(answer_form: dict) -> (str, list):
    idx_values = {i: [] for i in range(len(answer_form))}
    for _dict in answer_form:
        for i, v in enumerate(_dict.values()):
            idx_values[i].append(v.strip())

    candi_values = []
    for vs in idx_values.values():
        if len(set(vs)) > 1:
            return 'each', [remove_bracket(tmp_v) for tmp_v in vs]
        tmp_v = remove_bracket(vs[0])
        candi_values.append(tmp_v)
    return 'candi', candi_values


def remove_bracket(string):
    return re.sub('\([^\)]+\)', '', string)


def bracket_to_reg(string):
    reg_px = re.search('(\([^\)]+\))', string)
    if reg_px:
        reg = reg_px.groups()[0]
        return f'{string.replace(reg, "")}|{reg}'
    else:
        return string


def get_search_form(answer_form, answer_span):
    answer_blank_type = {}
    for tmp_item in answer_form:
        for k, v in tmp_item.items():
            if v.strip() == '':
                # 정답을 찾아야하는 key의 종류가 숫자인지 아닌지 확인
                answer_type = check_answer_type(k, answer_span)
                answer_blank_type[k] = answer_type

    search_form_list = []
    for _dict in answer_form:
        search_form = {}
        hints = []
        for k, v in _dict.items():

            # value가 있는 경우 그 value값을 구분자로 쓴다.
            # value가 비어있는 경우 찾아야하는 answer blank 이다.
            # remove_bracket(tmp_v)
            if v.strip() == '':
                answer_hint = []
                v_type = answer_blank_type[k]
                if v_type == 'char':
                    v = k.strip()
                elif v_type == 'kind':
                    key_k = remove_bracket(k)
                    v = f"{'|'.join(KIND_CHAR[key_k])}"
                else:
                    v = RG_DIGIT
                    answer_hint.append(remove_bracket(k))
                    tmp_r = re.search(RG_AFTER_DIGIT, answer_span)
                    if tmp_r and tmp_r.groups():
                        answer_hint.append(tmp_r.groups()[0])
                search_form[k] = v_type, hints + [
                    v
                ], f'({"|".join(answer_hint)})' if len(answer_hint) > 0 else ''
            else:
                n_v = remove_bracket(v.strip())
                hints.append(n_v)
        search_form_list.append(search_form)
    return search_form_list


def extract_value_from_output(answer_sent, answer_span, search_form_list):

    # remove output separators
    output_clean = answer_sent
    # output_clean = ''.join(answer_sent.split(' '))
    results = []
    for search_from in search_form_list:
        output_sub = output_clean
        for k, (v_type, v_pt, answer_form_hint) in search_from.items():
            result_from = {}
            pt_candis = []
            tmp_pt = f'({v_pt[-1]})'
            # output_sub = re.sub(tmp_pt, '_\g<1>', output_sub)
            if answer_form_hint:
                tmp_v_pt = copy.deepcopy(v_pt)
                tmp_v_pt[-1] = tmp_v_pt[-1] + f'(?={answer_form_hint})'
                # print('ph with hint: ', tmp_v_pt)
                pt_candis.append(
                    (0, f'({tmp_v_pt[-1]})' + '[^_]{,20}' + tmp_v_pt[0]))
                pt_candis.append(
                    (-1, tmp_v_pt[0] + '[^_]{,20}' + f'({tmp_v_pt[-1]})'))

            pt_candis.append((0, f'({v_pt[-1]})' + '[^_]{,20}' + v_pt[0]))
            pt_candis.append((-1, v_pt[0] + '[^_]{,20}' + f'({v_pt[-1]})'))

            for blank_idx, pt in pt_candis:
                rept = re.compile(pt)
                result = re.search(rept, output_sub)
                if not result:
                    continue
                else:
                    result_from[k] = result.groups()[0]
                    output_sub = output_sub[:result.start()] + re.sub(
                        result.groups()[blank_idx], '', output_sub[result.start(
                        ):result.end()]) + output_sub[result.end():]
                    break

            results.append(result_from)
    return results


def check_answer_type(key, answer_span):
    for kind_type in KIND_CHAR.keys():  # 계절, 요일 등 정해진 명사인 경우
        if kind_type in key:
            return 'kind'

    for digit_type in DIGIT_UNITS.keys():  # 정답이 숫자 단위인 경우
        if digit_type in key:
            return 'digit'

    if len(answer_span) < 15 and re.search(RG_DIGIT, answer_span):
        return 'digit'

    return 'char'  # 문양 등등