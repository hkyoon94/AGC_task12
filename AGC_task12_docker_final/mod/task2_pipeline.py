import time, json, jsbeautifier, sys; sys.path.append("../../../agc_main_project 2/src")
from glob import glob
from collections import OrderedDict
from sentence_transformers import util, SentenceTransformer
from xml.etree.ElementTree import XMLParser
from inference.tree_builder import CustomTreeBuilder
from inference.task1_routines import collect_high_cos_sim, multi_filter_and_cos
from inference.task2_routines import solve_task2_B, MULTI_FILTER, HIGH_COS
from models.mrc_infer2 import *
from const.task12_constant import mrc_model_path, encoder_path, is_local
from const.main_constant import data_path


def run(task_no_2_q, task_no_2_ans, doc_path):

    pipeline_arguments = {
        'task_no_2_q': task_no_2_q,
        'task_no_2_ans': task_no_2_ans,
        'doc_path': doc_path,

        'encoder': {
            'model': SentenceTransformer,
            'path': encoder_path
        },
        'mrc_model': {
            'path': mrc_model_path
        },
        'context_generator_task2': {
            'method': multi_filter_and_cos,
            'kwargs': {
                'around_num': 2,
            }
        },
        # 'context_generator_task2': {
        #     'method': collect_high_cos_sim,
        #     'kwargs': {
        #         'collect_num': 10,
        #         'ignore_short_sentences': True,
        #         'minimal_token_length': 10
        #         }
        # },
        'task2_solver': {
            'method': solve_task2_B,
            'kwargs': {}
        },
        'log_file_name': 'logs.json',
    }
    base_pipe = Pipeline(**pipeline_arguments)

    return base_pipe.task_no_2_ans


class Pipeline:
    def __init__(self,
                 task_no_2_q,
                 task_no_2_ans,
                 doc_path,
                 encoder,
                 mrc_model,
                 context_generator_task2,
                 task2_solver,
                 log_file_name,
                 trial_num=0):

        # 파이프라인 초기화-------------------------------------------------------------------------------------#
        print(f"\nInitializing trial {trial_num} ...\n")

        # 파이프라인 인수 정리
        self.task_no_2_q = task_no_2_q
        self.task_no_2_ans = task_no_2_ans
        self.doc_path = doc_path

        self.encoder = encoder['model'](encoder['path'])
        self.mrc_model = MRCModel(mrc_model['path'])
        self.qa_model = QAInference(self.mrc_model)
        self.mrc_model = MRCInference(self.mrc_model)

        self.generate_task2_context = context_generator_task2['method']
        self.task2_solver = task2_solver['method']

        self.log_file_name = log_file_name
        self.trial_num = trial_num
        self.task1_num = 0
        self.task2_num = 0
        self.task1_correct = 0
        self.task2_correct = 0
        self.task1_incorrect_list = []
        self.task2_incorrect_list = []
        self.task1_total_time = 0
        self.task2_total_time = 0

        self.correct_logs = []  # 로그 모음
        self.incorrect_logs = []
        self.pipeline_ct = -1  # 실행 수

        self.task2_log = []

        for prob_id, prob in enumerate(self.task_no_2_q):  # 지정한 문제들을 따라 순회
            try:
                print(f"Solving Prob. {prob['problem_no']} ...")
                self.pipeline_ct += 1
                self.incorrect_cause = []
                self.solving_progress = OrderedDict({})
                self.progress_issue_reports = []
                self.start_time = time.time()

                self.prob_no = prob['problem_no']  # 문제 번호
                self.task_no = prob['task_no']  # 문제 유형
                self.prob_query = prob['question']  # 문제의 질문
                self.answer_form = self.task_no_2_ans[prob_id]['answer']  # answer form

                if is_local:
                    # 데이터 repo 경로 설정 (xml_path 지정)
                    customdata_dir = os.path.join(data_path, "docs_filtered_json_v4")  # 수집 데이터 디렉토리 경로
                    self.doc_path = customdata_dir
                    self.xml_path = self.doc_path + prob["doc_file"] + '/content.xml'

                else:
                    xml_paths = glob(os.path.join(data_path + '/doc/*/*/content.xml'))
                    doc_file_name = prob['doc_file'].strip('.odt')
                    for xml_path in xml_paths:
                        if doc_file_name in xml_path:
                            self.xml_path = xml_path
                            break

                # tree_builder xml parser
                xml_to_json_parser = XMLParser(target=CustomTreeBuilder(title=prob["doc_file"]))

                # 전처리 루틴
                # xml -> json
                json_data = xml_to_json(self.xml_path, xml_to_json_parser)

                #!---------------------------------------- 코어 루틴 ---------------------------------------#
                # json (json_data) -> corpus (parsed_data) 의 메서드 콜
                corpus = parse_json_hierarchy(json_data=json_data)

                # 문제 해결
                # corpus, query -> context 생성 메서드 콜

                try:
                    query_contexts = self.generate_task2_context(self,
                                                                 corpus=corpus,
                                                                 query=self.prob_query,
                                                                 **context_generator_task2['kwargs'])

                    context_type = MULTI_FILTER
                    task2_contexts = query_contexts['contexts']['1']
                    
                except:
                    print('except!!')
                    context_type = HIGH_COS
                    task2_contexts = collect_high_cos_sim(
                                                            self,
                                                            corpus=corpus,
                                                            query=self.prob_query,
                                                            **{
                                                            'collect_num': 10,
                                                            'ignore_short_sentences': True,
                                                            'minimal_token_length': 10
                                                            }
                                                            )
                # task2_contexts = query_contexts
                # context_type = HIGH_COS
                inferred_answer = self.task2_solver(self,
                                                    context=task2_contexts,
                                                    query=self.prob_query,
                                                    corpus=corpus,
                                                    answer_form=self.answer_form,
                                                    context_type=context_type,
                                                    **task2_solver['kwargs'])
                self.task_no_2_ans[prob_id]['answer'] = inferred_answer
            except:
                continue

        if is_local:
            with open('task2_log.json', 'w', encoding='utf-8') as f:
                json.dump(self.task2_log, f, ensure_ascii=False)

    ## 파이프라인 끝


    ### pipeline 메서드

    def record_prob_time(self):
        self.prob_time = time.time() - self.start_time
        self.task1_total_time += self.prob_time

    def print_message(self):
        if self.answer_status == 'correct':
            print(f"Correct! ( {self.prob_time:.4f} sec )\n")
        else:
            print(f"Incorrect ... ( {self.prob_time:.4f} sec )\n")

    def check_task1_answer(self, inferred_answer):
        answer_similarity = util.pytorch_cos_sim(
            self.encoder.encode(inferred_answer, convert_to_tensor=True),
            self.encoder.encode(self.correct_answer, convert_to_tensor=True))
        if answer_similarity > 0.9:
            self.answer_status = 'correct'
            self.task1_correct += 1
        else:
            self.answer_status = 'incorrect'
            self.task1_incorrect_list.append(self.prob_no)
        self.task1_num += 1

    def check_task2_answer(self, inferred_answer):
        pass

    def write_prob_log(self):
        prob_log = OrderedDict({
            'problem_no':
                self.prob_no,
            'question':
                self.prob_query,
            'solving_progress':
                self.solving_progress,
            'progress_issues':
                self.progress_issue_reports,
            'correct_answer':
                self.correct_answer,
            'incorrect_cause':
                self.incorrect_cause,
            'status':
                self.answer_status,
            'inference_time':
                f'{self.prob_time:.4f} sec'
        })
        if self.answer_status == 'correct':
            self.correct_logs.append(prob_log)
        else:
            self.incorrect_logs.append(prob_log)


    def write_total_log(self):
        total_logs = []
        total_logs.append({'CORRECT PROBLEMS': 10 * "-----------"})
        total_logs.extend(self.correct_logs)
        total_logs.append({'INCORRECT PROBLEMS': 10 * "-----------"})
        total_logs.extend(self.incorrect_logs)
        total_logs.append({'SUMMARY': 10 * "-----------"})
        if not self.task1_num == 0:
            accuracy = 100 * self.task1_correct / self.task1_num
            total_logs.append({
                'task1_results': {
                    'accuracy':
                        (accuracy,
                         f"%, ( {self.task1_correct} out of {self.task1_num} )"
                        ),
                    'incorrect_no': self.task1_incorrect_list,
                    'total_elapsed_time': f"{self.task1_total_time:.4f} sec"
                }
            })
        else:
            total_logs.append({'task1_results': "None"})
        if not self.task2_num == 0:
            accuracy = 100 * self.task1_correct / self.task1_num
            total_logs.append({
                'task2_results': {
                    'accuracy':
                        (accuracy,
                         f"%, ( {self.task2_correct} out of {self.task2_num} )"
                        ),
                    'incorrect_no': self.task2_incorrect_list,
                    'total_elapsed_time': f"{self.task2_total_time:.4f} sec"
                }
            })
        else:
            total_logs.append({'task2_results': "None"})

        print(
            jsbeautifier.beautify(json.dumps(total_logs[-2],
                                             ensure_ascii=False)))
        print(
            jsbeautifier.beautify(json.dumps(total_logs[-1],
                                             ensure_ascii=False)))

        self.model_info.update({'trial_num': self.trial_num})
        total_logs.append({'model_info': self.model_info})
        print(
            jsbeautifier.beautify(json.dumps(total_logs[-1],
                                             ensure_ascii=False)))

        with open(self.log_file_name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(total_logs, indent=4, ensure_ascii=False))




### xml -> json 메서드
def xml_to_json(xml_path, parser, generate_json_file=False):
    with open(xml_path, 'r', encoding='utf-8') as f:
        test_data = f.read()
    parser.feed(test_data)
    json_data = parser.close()

    if generate_json_file is True:
        with open('xml_parsed.json', 'w', encoding='utf-8') as fp:
            fp.write(json.dumps(json_data, indent=4, ensure_ascii=False))

    return json_data

### json -> dict 메서드
def parse_json_hierarchy(json_data, **kwargs):
    parsed_data = []

    def recursive_picker(json_block):
        if json_block['type'] == 'text':
            parsed_data.append(json_block['value'])
        elif json_block['type'] == 'list':
            for subtag in json_block['value']:
                for j in json_block['value'][subtag]['value']:
                    recursive_picker(
                        json_block=json_block['value'][subtag]['value'][j])

    for i in json_data.keys():
        recursive_picker(json_block=json_data[i])

    return parsed_data

### text 정리 메서드
def cleanse(parsed_data):
    targets = ["□","◦","ㅇ","※","**","*","∙"]
    for key in parsed_data:
        for target in targets:
            parsed_data[key].replace(target, "")

    return parsed_data