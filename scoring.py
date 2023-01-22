import json
from AGC.task12_docker_final.mod import task1_pipeline
from const import main_constant

#답안 파일명
answer_sheet = 'answersheet.json'
#우리답안 파일명
ouranswer_sheet = 'our_answers.json'

#답안 불러오기
with open(main_constant.data_path + answer_sheet, "r", encoding="UTF-8") as f:
    answers = json.load(f)
sorted_answers = sorted(answers, key=lambda x: x["task_no"])

#우리답안 불러오기
with open(main_constant.data_path + ouranswer_sheet, "r", encoding="UTF-8") as f:
    ouranswers = json.load(f)
sorted_ouranswers = sorted(ouranswers, key=lambda x: x["task_no"])

#문젲지 불러오기
with open(main_constant.data_path + main_constant.problem_sheet, "r", encoding="UTF-8") as f:
    questions = json.load(f)
sorted_qa = sorted(questions, key=lambda x: x["task_no"])

#점수
score_task1 = 0
score_task2 = 0
score_task3 = 0
score_task4 = 0

for ourans in sorted_ouranswers:
    problem_no= ourans.get('problem_no')
    task_no = ourans.get('task_no')

    #답에 해당하는 문제 찾기
    for q in sorted_qa:
        if q['problem_no'] == problem_no:
            tmp_q = q
    #현재 문제의 실제 답 구하기
    tmp_ans = ''
    for ans in sorted_answers:
        if ans.get('problem_no') == problem_no:
            tmp_ans = ans

    #task 별 scoring
    if task_no == "1":
        our_tmp_ans_opts = ourans.get('answer')
        for tmp_ans_opt in our_tmp_ans_opts:
            if tmp_ans_opt.get('paragraph') == tmp_ans.get('answer'):
                score_task1 += round(1/tmp_ans_opt.get('rank'),2)

    elif task_no == "2":
        our_tmp_ans_list = ourans.get('answer')


    elif task_no == "3":
        tmp_ans = ans.get('answer')
    elif task_no == "4":
        tmp_ans = ans.get('answer')
