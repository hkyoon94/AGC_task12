import copy
import json
import re
# import kss

RG_DIGIT = r'[0-9]+([\.\,][0-9]+)?'


def cnt_loc_diff(str1, str2, str1_loc, str2_loc):
    overlapped = 0
    if str1_loc < str2_loc:
        if str1_loc + len(str1) >= str2_loc:
            return overlapped
        return str2_loc - str1_loc - len(
            str1)  # key index 뒤에 정답이 등장했을 때 (str1) (str2)
    elif str1_loc > str2_loc:  # key index 앞에 정답이 등장했을 때 (str2) (str1)
        if str2_loc + len(str2) >= str1_loc:
            return overlapped
        return str1_loc - str2_loc - len(str2)
    else:
        return overlapped


def extract_digit(string):
    digit = re.search(RG_DIGIT, string)
    if digit:
        return digit.group()
    else:
        return None


def get_answer_value_candis(answers):
    ans_candis = []
    for i, ans in enumerate(answers):
        matches = re.finditer(RG_DIGIT, ans)
        if matches:  # digit match
            ans_candi = [d.group() for d in matches]
            if len(ans_candi) > 1:
                ans_candis += ans_candi
            else:
                ans_candis += [ans]
            print([d.group() for d in re.finditer(RG_DIGIT, ans)])

        else:
            # TODO: 형태소 분석 이용하여 조사/부사 제거하기
            ans_candis.append(ans)
    return ans_candis


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1:
            return
        yield start
        start += len(sub)


def get_answer_form_hint(answer_form):
    keys = []
    for r in answer_form:
        key = []
        for j, (k, v) in enumerate(r.items()):
            if v.strip() != '':
                key.append(v)
            else:
                if j == 0:
                    key.append(k)
        keys.append(key)
    return keys


def count_hint_from_context(hint, context):
    count_by_key_lots = []
    for j, c in enumerate(context):
        # 모든 key hint가 있는지 확인하기
        every_key_is_here = 0
        for k in hint:  # hint가 여러개인경우 every_key_is_here는 2이상이 될 수 있음
            if re.search(''.join(k.split()), ''.join(c.split())):
                every_key_is_here += 1
        count_by_key_lots.append((j, copy.deepcopy(every_key_is_here)))
    return sorted(count_by_key_lots, key=lambda x: x[1], reverse=True)


def map_key_value_from_context(answers, answer_form, context):
    ans_candis = get_answer_value_candis(answers)
    keys = get_answer_form_hint(answer_form)
    new_context = context
    ##### 각 key 별로 context를 돌며 많은 key hint가 있는 것에서 answer value를 맵핑해본다.
    for i, key in enumerate(keys):
        ##### 모든 context에 hint 개수 서치
        sorted_list = count_hint_from_context(hint=key, context=new_context)
        locs = []

        for c_idx, key_cnt in sorted_list:  # hint가 많이 등장한 context 순으로 keyhint와 가장 가까운 ans를 찾는다.
            # 마지막 key hint의 위치를 저장한다.
            if key_cnt == 0:
                break
            # key_loc = new_context[c_idx].find(key[-1])
            # TODO 등장하는 key들 다 추출하기 = > mrc의 위치를 전적으로 믿어야함. mrc 위치 주변에서 찾기
            main_hint = key[-1].replace(' ', '\s?')
            key_locs = [
                m.start() for m in re.finditer(main_hint, new_context[c_idx])
            ]
            for key_loc in key_locs:
                for ac_idx, ac in enumerate(ans_candis):
                    ac_matches = find_all(new_context[c_idx], ac)
                    for loc in ac_matches:
                        diff = cnt_loc_diff(key[-1], ac, key_loc, loc)
                        if diff == -100:
                            continue
                        locs.append((diff, ac_idx, c_idx, loc))
            # if len(sorted_list) - 1 == c_idx:
            #     print(c_idx)
            # 모든 리스트 끝까지 오거나 동일 hint 개수가 끝날 때 거리비교
            if len(sorted_list) - 1 == c_idx or key_cnt != sorted_list[c_idx +
                                                                       1][1]:
                ordered = sorted(locs, key=lambda x: x[0])
                if len(ordered) > 0 and ordered[0][0] < 40:
                    for k, v in answer_form[i].items():
                        if v.strip() == '':
                            digit = extract_digit(ans_candis[ordered[0][1]])
                            if digit:
                                answer_form[i][k] = digit
                            else:
                                answer_form[i][k] = ans_candis[ordered[0][1]]
                else:
                    locs = []
    return answer_form


###########################################################################################
# List Pattern
###########################################################################################

RG_CHAR_COMMA_LIST = r'[^및\n]{1,20}및[^,\n]{1,20}(,[^,\n]{1,20}){1,}$'
RG_COMMA_LIST = r'[^,\n]{1,20}(,[^,\n]{1,20}){1,},?$'
RG_CHAR_LIST = r'.{1,20}([와과](\s)|및).{1,20}$'

RG_SCHAR = '[·▲▴]'
RG_SCHAR_LIST = r'[^·▲▴\n]{1,8}([·▲▴][^·▴▲\n]{1,10}){2,}$'

RG_CLEAN_CHOSA = r'[을를]\s?$'
RG_CLEAN_DEPEN_NOUN = r'(\s)등(에|\s)?$'


def get_list_pattern(string):
    comma_ptt = re.search(RG_COMMA_LIST, string)
    char_ptt = re.search(RG_CHAR_LIST, string)
    schar_ptt = re.search(RG_SCHAR_LIST, string)
    comma_char_ptt = re.search(RG_CHAR_COMMA_LIST, string)
    if comma_char_ptt:
        string = string.split('및')
        sub_char = re.sub('(?![0-9]),(?![0-9])', '++', string[-1])
        result = string[:-1] + sub_char.split('++')
    elif comma_ptt:
        string = re.sub('(?![0-9]),(?![0-9])', '++', string)
        result = string.split('++')
    elif char_ptt:
        string = re.sub(r'([와과](\s)|및)', '++', string)
        result = string.split('++')
    elif schar_ptt:
        string = re.sub(RG_SCHAR, '++', string)
        result = string.split('++')
    else:
        result = []

    for i, r in enumerate(result):
        comma_ptt = re.search(RG_COMMA_LIST, r)
        if comma_ptt:
            string = re.sub('(?![0-9]),(?![0-9])', '++', r)
            del result[i]
            result += string.split('++')
    return result


def clean_output_list(string):
    cleaned_string = string.strip()
    pt_clean_num = re.search('^[①②③④]+', cleaned_string)
    pt_clean_schar_str = re.search('^\W+', cleaned_string)
    pt_clean_schar_end = re.search('\W+$', cleaned_string)
    pt_clean_chosa = re.search(RG_CLEAN_CHOSA, string)
    pt_clean_dnoun = re.search(RG_CLEAN_DEPEN_NOUN, string)
    if pt_clean_schar_str:
        cleaned_string = re.sub('^\W+', '', cleaned_string)
    if pt_clean_schar_end:
        cleaned_string = re.sub('\W+$', '', cleaned_string)
    if pt_clean_chosa:
        cleaned_string = re.sub(RG_CLEAN_CHOSA, '', cleaned_string)
    if pt_clean_dnoun:
        cleaned_string = re.sub(RG_CLEAN_DEPEN_NOUN, '', cleaned_string)
    if pt_clean_num:
        cleaned_string = re.sub('^[①②③]+', '', cleaned_string)
    return cleaned_string


def count_answer_from_context(hint, context):
    count_by_key_lots = []
    for j, c in enumerate(context):
        # 모든 key hint가 있는지 확인하기
        every_key_is_here = 0
        for i, k in enumerate(
                hint):  # hint가 여러개인경우 every_key_is_here는 2이상이 될 수 있음
            if find_all(''.join(k.split()), ''.join(c.split())):
                every_key_is_here += 1
        count_by_key_lots.append((j, copy.deepcopy(every_key_is_here)))
    return sorted(count_by_key_lots, key=lambda x: x[1], reverse=True)


def map_list_from_context(answers, answer_form, context):
    """
    answer_form이 list 형식일때 mapping rule
         - answers: mrc infer result
         - answer_form: form to be filled
         - context: extracted context from doc by extractor
        # ans_candis 추출
    1) mrc 답에 나열 형식이 있는 경우, 각 element 들을 추출한다.
        예) ,,, or ~(와/과/나)

    # key hint 추출
    1) answer_form 의 key값을 추출한다.

    # 후처리
    1) 앞 뒤의 special character 제거
    2) \s등, 가/는 등의 조사 제거

    # 알고리즘  A
    1. 리스트 형식인지
        - char+(, char{,20}){2} 들이 3개이상 반복
        - (와|과)\s
        -
    2. anser candi 들이 많이 있어야함
    3. 요소들의 개수가 anser_form 이랑 유사한지

    * 수도코드
    1) mrc_answer들이 있는 문단 idx을 추출한다.
    2) 많은 순서대로 sorting
    3) for loop
    for answer in mrc_answer:
        for c in ordered_context:
            if answer 포함하는 영역에서 list 있는지 확인
                answer 포함하는 영역에서 list 추출
            else:
                continue
    if 추출된 결과가 문항 수보다 적은 경우
        겹치는 것 제외하고 answer 에서 뽑아낸다.

    # 알고리즘 B
    MRC 결과들을 위주로 본다.
    1) if 리스트 형식의 answer 들을 찾아본다.
        -> 있다
            리스트 형식을 잘라 answer form에 추가
            if answer form이 모두 채워지지 않았다 -> 2로직 실행
        -> 없다
        2) 로직 실행
    2) context에서 answer candi 들이 가장 많이 등장한 c 부터 서치 시작
       - answer 들 포함하는 영역에서 list 있는지 확인
       -

    """
    # 알고리즘 B
    answer_candis = []
    candis_idx = []
    for idx, ans in enumerate(answers):
        elements = get_list_pattern(ans)
        if elements:
            candis_idx.append(idx)
            answer_candis += elements
    print(answer_candis)
    print(answer_form)
    if len(answer_candis) > 0:
        for i, dict_ in enumerate(answer_form):
            for k, v in dict_.items():
                if v.strip() == '':
                    if len(answer_candis) > i and answer_candis[i].strip():
                        answer_form[i][k] = clean_output_list(
                            answer_candis[i].strip())
    print(answer_form)
    # answer_candis + mrc_answers 가 가장 많이 등장한 context search
    # 단, answer_candis로 나눠진 element 제외,
    # 단,
    tmp_answer = [e for i, e in enumerate(answers) if i not in candis_idx]
    hint = tmp_answer + answer_candis
    sorted_list = count_answer_from_context(hint=hint, context=context)

    processed_idx = []
    for idx, c in enumerate(sorted_list):
        if idx in processed_idx:
            continue
        else:
            processed_idx.append(idx)
        sent = context[idx]
        print(sent)
        # start = find_all(sent, answers[i])

    vacant_idx = 0
    for i, dict_ in enumerate(answer_form):
        for k, v in dict_.items():
            if v.strip() == '':
                answer_form[i][k] = clean_output_list(tmp_answer[vacant_idx])
                vacant_idx += 1
    return answer_form