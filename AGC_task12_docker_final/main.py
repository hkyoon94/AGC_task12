import os
import json
from urllib import request

# from mod import task4_pipeline
# from mod import task3_pipeline
from mod import task1_pipeline, task2_pipeline
from const import main_constant

# def send_ans(answers, api_url):
#     for ans in answers:
#         # json dump & encode utf-8
#         tmp_message = json.dumps(ans).encode('utf-8')
#         request_message = request.Request(api_url, data=tmp_message)
#         resp = request.urlopen(request_message)  # POST

#         resp_json = eval(resp.read().decode('utf-8'))
#         print("received message: " + resp_json['msg'])

#         if "OK" == resp_json['status']:
#             print("data requests successful!!")
#         elif "ERROR" == resp_json['status']:
#             raise ValueError("Receive ERROR status. Please check your source code.")

# def main():
# load environment variable
#api_url = os.environ['REST_ANSWER_URL']
#data_path = "/home/agc2022/dataset/" #실제 path

# api_url = 'http://restapi.example.com/users/post-comments'
# data_path = "../dataset/"
problem_sheet = "/problemsheet.json"
team_id = "bsj1009"
hash_value = "aGGsvA6FFyfSs0QT"


with open(main_constant.data_path + problem_sheet, "r", encoding="UTF-8") as f:

    questions = json.load(f)

sorted_qa = sorted(questions, key=lambda x: x["task_no"])

task_no_1_q = []
task_no_2_q = []
task_no_3_q = []
task_no_4_q = []
task_etc_q = []

task_no_1_ans = []
task_no_2_ans = []
task_no_3_ans = []
task_no_4_ans = []

# 문제 및 답안 생성
for q in sorted_qa:
    tmp_ans_sheet = {
        "team_id": team_id,
        "hash": hash_value,
        "problem_no": q["problem_no"],
        "task_no": q["task_no"]
    }
    if q["task_no"] == "1":
        task_no_1_q.append(q)
        tmp_ans_sheet["answer"] = [
            {
                "rank": "1",
                "paragraph": ""
            },
            {
                "rank": "2",
                "paragraph": ""
            },
            {
                "rank": "3",
                "paragraph": ""
            }
        ]
        task_no_1_ans.append(tmp_ans_sheet)
    elif q["task_no"] == "2":
        task_no_2_q.append(q)
        tmp_ans_sheet["answer"] = q["answer_form"]
        task_no_2_ans.append(tmp_ans_sheet)
    elif q["task_no"] == "3":
        task_no_3_q.append(q)
        tmp_ans_sheet["answer"] = ""
        task_no_3_ans.append(tmp_ans_sheet)
    elif q["task_no"] == "4":
        task_no_4_q.append(q)
        tmp_ans_sheet["answer"] = ""
        tmp_ans_sheet["evidence"] = []
        task_no_4_ans.append(tmp_ans_sheet)
    else:
        task_etc_q.append(q)
        print("task 유형이 없음")



#각데이터 파이프라인을 호출 후, 답안 return
#task12 pipeline 시작
task_no_1_ans = task1_pipeline.run(task_no_1_q, task_no_1_ans, main_constant.doc_path)
task_no_2_ans = task2_pipeline.run(task_no_2_q, task_no_2_ans, main_constant.doc_path)

with open("our_answers.json", 'w', encoding='utf-8') as f:
    f.write(json.dumps(task_no_1_ans+task_no_2_ans,indent=4,ensure_ascii=False))
    
# send_ans(answers=task_no_12_ans,api_url=api_url)

    #task3 pipeline 시작
    # task_no_3_ans = task3_pipeline.run(task_no_3_q, task_no_3_ans, main_constant.ref_path)
    # send_ans(answers=task_no_3_ans,api_url=api_url)


    #task4 pipeline 시작
    # reader_model_path = './models/task4_reader_checkpoint'
    # passage_tokenizer_path = './task4/passage_generation_tokenizer'
    # tmp_task_no_4_ans = task4_pipeline.solve(task_no_4_q, main_constant.doc_path, reader_model_path, passage_tokenizer_path)
    # for ans in tmp_task_no_4_ans:
    #     ans["task_no"] = '4'
    #     ans["team_id"]=team_id
    #     ans["hash"]=hash_value
    # send_ans(answers=tmp_task_no_4_ans, api_url=api_url)

    # # request end of mission message
    # message_structure = {
    #     "team_id": team_id,
    #     "hash": hash_value,
    #     "end_of_mission": "true"
    # }
    # #답안에 끝났다는 메시지 추가
    # send_ans(answers=[message_structure], api_url=api_url)
    # print('여기까지 오니?')


# if __name__ == "__main__":
#     main()