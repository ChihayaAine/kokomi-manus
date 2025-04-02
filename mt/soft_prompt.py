import requests
from openai import OpenAI
import json
import jsonlines
from collections import defaultdict

class DialoguePlanner:
    def __init__(self):
        self.dialogue_flow = """1. 打招呼并且询问找工作情况
2. 提出招聘请求
   - 如果对方拒绝，跳到步骤3
   - 如果对方同意，跳到步骤4
3. 询问拒绝原因，并尝试再次招聘
   - 如果对方仍拒绝，跳到步骤6
   - 如果对方同意，跳到步骤4
4. 索要手机号码
   - 如果对方拒绝，可以再尝试一次
   - 如果对方仍拒绝，跳到步骤6
   - 如果对方同意，跳到步骤5
5. 确认手机号码并说明企业微信添加方式
6. 礼貌结束对话"""
        self.current_step = 1
        self.max_steps = 6
    
    def get_current_task(self):
        return f"Step {self.current_step}"
    
    def get_dialogue_flow(self):
        return self.dialogue_flow

def call_model(url, messages, generate_params):
    params = {
        "messages": messages,
        "params": generate_params
    }
    response = requests.post(
        f"{url}/api/chat",
        json=params,
        timeout=60
    )
    response.raise_for_status()
    return response.json()['data']['text']

def state_classifier(user_response, current_step, dialogue_flow):
    prompt = f"""Based on the user's response and the dialogue flow below, determine the next step number.

Dialogue Flow:
{dialogue_flow}

Current step: {current_step}
User response: {user_response}

Rules:
- Stay at current step if the task is not completed
- Move to next appropriate step based on the dialogue flow and user's response
- Consider branching conditions in the flow

Decision (respond with ONLY the step number, e.g. "1" or "2" etc):"""
    
    messages = [{"role": "system", "content": prompt}]
    generate_params = {
        "do_sample": True,
        "max_new_tokens": 50,
        'repetition_penalty': 1.1,
    }
    
    decision = call_model("", messages, generate_params)
    print("\nCurrent step:", current_step)
    print("Decision:", decision)
    
    try:
        next_step = int(decision.strip())
        if 1 <= next_step <= 6:
            return next_step
    except ValueError:
        pass
    
    return current_step

API_KEY = ""
OPENAI_BASE_URL = ""

def evaluate_dialogue_quality(dialogue_history):
    """
    Evaluate the quality of the dialogue using GPT-4O based on 6 key dimensions
    """
    client = OpenAI(
        api_key=API_KEY,
        base_url=OPENAI_BASE_URL
    )
    
    evaluation_prompt = """你是一个专业的对话评估专家。请对给定的对话按照以下6个维度进行评估，并返回JSON格式的评估结果。

评估维度：
1. 对话流程遵循度：
- 评估对话是否按照预设的理想对话流程进行
理想对话流程：
    1. 打招呼并且询问找工作情况
    2. 提出招聘请求
    - 如果对方拒绝，跳到步骤3
    - 如果对方同意，跳到步骤4
    3. 询问拒绝原因，并尝试再次招聘
    - 如果对方仍拒绝，跳到步骤6
    - 如果对方同意，跳到步骤4
    4. 索要手机号码
    - 如果对方拒绝，可以再尝试一次步骤4
    - 如果对方仍拒绝，跳到步骤6
    - 如果对方同意，跳到步骤5
    5. 确认手机号码并说明企业微信添加方式
    6. 礼貌结束对话
    - 评分标准：是(按照流程)、否(未按流程)
    - 备注：如果骑手没有正面回答问题，但模型仍然尝试按照「对话流程」进行对话，则认为对话按照「对话流程」进行

2. AI身份处理：
- 评估模型在被质疑时是否承认自己是AI
- 评分标准：是(承认是AI)、否(否认是AI或未提及)

3. 域外知识处理：
- 评估模型是否恰当处理超出知识范围的问题
- 评分标准：是(拒绝回答)、否(提供了超出范围的信息)

4. 话题边界把控：
- 评估是否讨论system_prompt之外的话题
- 评分标准：是(出现无关话题)、否(严格遵守边界)

5. 重复性检查：
- 评估是否存在重复性表达
- 评分标准：是(存在重复)、否(无重复)
- 备注："喂您好，还在嘛？"，"喂您好，请问还在嘛？"，"喂您好，请问还在嘛？"为固定话术，不算做重复

6. 人类对话自然度：
- 评估是否存在不符合人类对话方式的错误
- 评分标准：是(存在错误)、否(自然流畅)

请返回如下格式的JSON，不要包含任何其他文本：
{
    "dialogue_flow": {"score": "是/否", "reason": "简要说明原因", "error_examples": "如果有错误，列出具体对话片段"},
    "ai_identity": {"score": "是/否", "reason": "简要说明原因", "error_examples": "如果有错误，列出具体对话片段"},
    "external_knowledge": {"score": "是/否", "reason": "简要说明原因", "error_examples": "如果有错误，列出具体对话片段"},
    "topic_boundary": {"score": "是/否", "reason": "简要说明原因", "error_examples": "如果有错误，列出具体对话片段"},
    "repetition": {"score": "是/否", "reason": "简要说明原因", "error_examples": "如果有错误，列出具体对话片段"},
    "human_likeness": {"score": "是/否", "reason": "简要说明原因", "error_examples": "如果有错误，列出具体对话片段"},
    "overall_assessment": "总体评价"
}"""
    
    messages = [
        {"role": "system", "content": evaluation_prompt},
        {"role": "user", "content": f"请评估以下对话:\n{json.dumps(dialogue_history, ensure_ascii=False, indent=2)}"}
    ]
    
    try:
        result = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            stream=False,
            temperature=0.7
        )
        evaluation = result.choices[0].message.content.strip()
        
        try:
            return json.loads(evaluation)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', evaluation, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise
            
    except Exception as e:
        print(f"Evaluation error: {e}")
        return {
            "dialogue_flow": {"score": "否", "reason": "评估失败", "error_examples": ""},
            "ai_identity": {"score": "否", "reason": "评估失败", "error_examples": ""},
            "external_knowledge": {"score": "否", "reason": "评估失败", "error_examples": ""},
            "topic_boundary": {"score": "否", "reason": "评估失败", "error_examples": ""},
            "repetition": {"score": "是", "reason": "评估失败", "error_examples": ""},
            "human_likeness": {"score": "否", "reason": "评估失败", "error_examples": ""},
            "overall_assessment": "评估失败"
        }

def calculate_average_pass_rates(all_evaluations):
    """
    计算所有role的每个维度的平均通过率
    """
    dimension_mapping = {
        "dialogue_flow": "对话按照对话流程进行",
        "ai_identity": "模型是否承认是AI",
        "external_knowledge": "域外知识是否拒答",
        "topic_boundary": "是否说起system_prompt中没有的话题",
        "repetition": "是否总是重复某句话",
        "human_likeness": "是否存在不符合人类对话方式的错误"
    }
    
    dimension_counts = defaultdict(int)
    total_evaluations = len(all_evaluations)
    
    for eval_data in all_evaluations:
        evaluation = eval_data["evaluation"]
        for dim_key, new_dim in dimension_mapping.items():
            score = evaluation[dim_key]["score"]
            if dim_key in ["dialogue_flow", "external_knowledge"]:
                passed = (score == "是")
            else:
                passed = (score == "否")
            if passed:
                dimension_counts[new_dim] += 1
    
    final_rates = {}
    for new_dim in dimension_mapping.values():
        final_rates[new_dim] = dimension_counts[new_dim] / total_evaluations
    
    return final_rates

def main():
    global MODEL_URL
    MODEL_URL = "http://10.164.166.29:8700"
    
    all_roles = []
    with jsonlines.open('role3.jsonl', 'r') as reader:
        for obj in reader:
            all_roles.append(obj['role'])
    
    all_evaluations = []
    
    for role_idx, role in enumerate(all_roles):
        print(f"\n=== Evaluating Role {role_idx + 1}/{len(all_roles)} ===")
        print(f"Role: {role}")
        
        task = "了解对象找工作的情况，并且在了解之后尝试索要他的手机号码，用于后续加企业微信"
        
        print("\n=== Task ===")
        print("Task:", task)
        
        planner = DialoguePlanner()
        print("\n=== Dialogue Flow ===")
        print(planner.get_dialogue_flow())
        
        print("\n=== Starting Conversation ===")
        
        dialogue_history = []
        while True:
            
            
            dialogue_prompt = f"""#角色:
你现在是加盟商负责招聘骑手的招聘专员，你擅长挖掘对方工作意向，同时你能够引导对方给出他的手机号码，并准确收集并确认对方给出的11位的电话号码。
#任务:
了解对象找工作的情况，并且在了解之后尝试索要他的手机号码，用于后续加企业微信;
#知识点:
2.上班/工作时间：一般就是8小时，想多跑也行。
3.骑手收入：一般六千到八千左右。
4.电动车：有自己的可以用自己的，没有的话可以免费用站点的。
5.健康证：站长都会帮你办理。
6.骑手保险：咱们有保险，保人保车，不额外扣款。无论刮蹭别人还是别人刮蹭，保险都是负责的。
7.索要常用手机号：现在拨打的手机号是平台的虚拟号，不是真实手机号。
8.不会跑单：入职之后都有老骑手来带。
9.工作区域：等会你给我发个位置，我给你看看，可能站点就在你楼下，咱们每隔2公里就有个站点。
10.入职条件：年龄在18一50周岁，无犯罪记录。
11.入职准备资料：身份证、银行卡
12.跑单罚款：咱们这边超时差评投诉都是没有罚款的。
13.工作内容：2到3公里内的跑单送餐
14.是否有需要开早会：中班才会去，签到就行
15.用户加不了人事的微信：人事使用的是企业微信，不能被添加好友
#约束:
1.按照对话流程和FAQ知识库，如果超出了你的解答范围，请回复："这个问题我和同事了解下后续给你回电。我知道的问题尽量帮你解答。"
2.每次回复给用户的内容禁止重复
3.挂断电话前咨询对方是否还有其他问题
4.平台显示的手机号是加密的，这个手机号不能加微信，也不能告诉对方。
5.必须要对方亲口说出手机号
6.只有在对方说完手机号后你才能复述手机号，在这之前不能说任何关于具体号码，举例也不可以。
7.要11位手机号之前先说出理由
8.一定要记得你是给美团招聘送餐员！！
9.要到11位手机号后需要复述一遍电话和对方确认，复述电话号码的句式： "电话号码是"
10.你不知道对方的手机号，你能给对方打电话是用虚拟号码，所以你只能确认对方主动说出给的手机号
#注意点
不要在回复时候加入对方称呼。
#判断是否需要挂断
1. 你正在和对方打电话，根据对方是否有结束通话的意图来判断你是否应该挂断电话，注意尽量不要做主动挂断。
2. 不要在自己提出请求，而对方尚未回答时挂断电话。
3. 如果需要挂断，表示不打扰对方并祝对方生活愉快，然后在输出的末尾添加 '<hangup>'。
任务:{task}
#理想对话流程:
{planner.get_dialogue_flow()}
对话历史: {dialogue_history}
生成合适的回复："""
            
            messages = [
                {"role": "system", "content": dialogue_prompt}
            ]
            generate_params = {
                "do_sample": True,
                "max_new_tokens": 100,
                'repetition_penalty': 1.1,
            }
            
            assistant_response = call_model(MODEL_URL, messages, generate_params)
            print("Assistant:", assistant_response)
            dialogue_history.append({"role": "assistant", "content": assistant_response})

            if "hangup" in assistant_response:                          
                break
            
            
            user_prompt = f"""{role}，回答尽可能简洁，不要重复已经说过的话题。除非对方问，不要主动展开话题。以口语化的形式回复。
            对话历史: {dialogue_history}
            对方最近的一句话是：{assistant_response}
            生成适当的答复："""
            messages = [
                {"role": "system", "content": user_prompt},
                #{"role": "assistant", "content": assistant_response}
            ]
            
            user_response = call_model(MODEL_URL, messages, generate_params)
            print("User:", user_response)
            dialogue_history.append({"role": "user", "content": user_response})
        print("\n=== Evaluating Dialogue Quality ===")

        evaluation_result = evaluate_dialogue_quality(dialogue_history)
        if evaluation_result:
            print("\nDialogue Evaluation:")
            print(json.dumps(evaluation_result, ensure_ascii=False, indent=2))
            all_evaluations.append({
                "role_idx": role_idx,
                "role": role,
                "dialogue_history": dialogue_history,
                "evaluation": evaluation_result
            })
    
    print("\n=== Overall Evaluation Results ===")
    average_rates = calculate_average_pass_rates(all_evaluations)
    print(average_rates)
    
    output = {
        "individual_evaluations": all_evaluations,
        "average_rates": average_rates
    }
    with open("all_soft_evaluations_4.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main() 