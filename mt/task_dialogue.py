import json
from collections import defaultdict
import requests
from task_dag import parse_dag_file, get_start_end_nodes
from tqdm import tqdm

class DialoguePlanner:
    def __init__(self, graph, node_labels):
        self.graph = graph
        self.node_labels = node_labels
        start_nodes, _ = get_start_end_nodes(graph)
        self.current_node = start_nodes[0]  # 假设只有一个起始节点
    
    def get_current_task(self):
        """获取当前节点对应的任务描述"""
        return self.node_labels.get(self.current_node, "")
    
    def get_next_nodes(self):
        """获取当前节点的所有可能下一个节点"""
        return self.graph.get(self.current_node, [])
    
    def move_to_node(self, next_node):
        """移动到新节点"""
        if next_node in self.get_next_nodes() or next_node == self.current_node:
            self.current_node = next_node
            return True
        return False

def call_model(url, messages, generate_params):
    """调用模型API"""
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

def get_task_decomposition(task):
    """调用模型分解任务为DAG"""
    decomposition_prompt = """Task:[task]
Design a directed acyclic task flow suitable for visualization using mermaid.js, taking into account the context of Task. This directed acyclic graph should decompose the entire TASK into multiple subtasks and express their dependencies using directed edges to describe the entire task flow. Generate only directed acyclic graphs without additional explanation. Please follow the following guidelines:
Node definition: Use different nodes to represent subtasks.Nodes must be used to represent the status of assistants rather than user.Nodes MUST be named exactly as "N1", "N2", "N3", "N4", etc. (Do NOT use A, B, C or any other format, and do NOT add any prefixes before N1).
Example format:
    N1[Start: Greet User and Ask About Job Search] --> N2[Request Phone Number for Follow-Up]
    N1 --> N3[Engage in Casual Chat to Ease Tension]

    N2 --> N4[Thank User and Confirm Addition to Enterprise WeChat]
    N4 --> N5[End Interaction]

    N3 --> N6[Repropose Recruitment]
    N6 --> N7[Politely End Interaction]
    N6 --> N2
Advanced dialogue operation: Each node should encapsulate the core emotion or function of the fragment in a directed acyclic task flow, related to Task. It should be a label representing nodes of advanced dialogue actions, not just dialogues.
Flow and directionality: Create directed connections between nodes to represent the progress and dependencies of subtasks. The task flow should flow from one node to potentially multiple nodes.
Subtask relationship: If multiple subtasks must be completed in their entirety, they will be arranged in series; If multiple subtasks are selected to be completed, they will be arranged in parallel, and selecting one of the subtasks will lead to different subtask paths (but different paths may point to the same node in the future).
Non cyclic structure: The dialogue flow must not have loops or cyclic paths.
Mermaid.js compatibility: Ensure that the built stream follows Mermaid.js graphical symbols to ensure seamless presentation.
Keep the number of subtask nodes concise and manageable (preferably between 5-10 nodes).
Taking into account these guidelines, develop a task process with a focus on Task. Tasks always start with greeting users and asking them what they want. The diagram should be connected.Don't add useless prefixes, just nodes and edges."""
    
    messages = [
        {"role": "system", "content": decomposition_prompt.replace("[task]", task)},
        {"role": "user", "content": "Please decompose the task into a DAG."}
    ]
    
    # 这里使用与dialog.py相同的生成参数
    generate_params = {
        "do_sample": True,
        "max_new_tokens": 300,
        'top_k': 1,
        'num_beams': 1,
        'repetition_penalty': 1.1,
    }
    
    # 调用模型获取分解结果
    dag_text = call_model(MODEL_URL, messages, generate_params)
    
    # 将结果写入临时文件供task_dag.py解析
    with open("temp_dag.txt", "w") as f:
        f.write(dag_text)
    
    return "temp_dag.txt"

def state_classifier(user_response, current_node, next_nodes, node_labels):
    """模型C：决策下一个状态"""
    prompt = f"""Based on the user's response, decide whether to:
1. Stay at current node (if the current task is not completed)
2. Move to next node (specify which one)

Current node task: {node_labels[current_node]}
Possible next tasks: {[node_labels[node] for node in next_nodes]}
User response: {user_response}

Rules:
- Choose option 1 if the current task needs more interaction
- Choose option 2 if current task is done and we should move forward

Decision:"""
    
    messages = [
        {"role": "system", "content": prompt}
    ]
    
    generate_params = {
        "do_sample": True,
        "max_new_tokens": 50,
        'repetition_penalty': 1.1,
    }
    
    decision = call_model(MODEL_URL, messages, generate_params)
    
    print("\nCurrent node:", current_node)
    print("Available next nodes:", next_nodes)
    print("Decision:", decision)
    
    # 检查是否需要结束对话
    if 'END_CONVERSATION' in decision:
        return 'END_CONVERSATION'
    
    # 解析决策
    if "stay" in decision.lower() or "option 1" in decision.lower():
        print("Staying at current node:", current_node)
        return current_node
        
    if "move" in decision.lower() or "option 2" in decision.lower():
        # 如果有多个下一节点，需要从决策文本中识别具体选择哪个
        for node in next_nodes:
            if node_labels[node].lower() in decision.lower():
                print("Moving to node:", node)
                return node
        # 如果只有一个下一节点，直接选择它
        if len(next_nodes) == 1:
            print("Moving to only available next node:", next_nodes[0])
            return next_nodes[0]
    
    # 如果无法做出明确决策，保持在当前节点
    print("Unable to make clear decision, staying at:", current_node)
    return current_node

def main():
    # 初始化配置
    global MODEL_URL
    MODEL_URL = ""
    
    # 示例任务
    task = "了解对象找工作的情况，并且在了解之后尝试索要他的手机号码，用于后续加企业微信"
    
    # 1. 任务分解
    print("\n=== Task Decomposition ===")
    print("Task:", task)
    
    dag_file = get_task_decomposition(task)
    
    # 打印生成的DAG文本
    print("\n=== Generated DAG ===")
    with open(dag_file, 'r') as f:
        print(f.read())
    
    # 解析DAG
    graph, node_labels = parse_dag_file(dag_file)
    
    # 打印解析后的数据结构
    print("\n=== Parsed DAG Structure ===")
    print("Graph (edges):", dict(graph))
    print("\nNode labels:", node_labels)
    
    start_nodes, end_nodes = get_start_end_nodes(graph)
    print("\nStart nodes:", start_nodes)
    print("End nodes:", end_nodes)
    
    print("\n=== Starting Conversation ===")
    
    # 初始化planner并开始对话
    planner = DialoguePlanner(graph, node_labels)
    
    # 2. 开始对话循环
    dialogue_history = []
    while True:
        current_task = planner.get_current_task()
        next_nodes = planner.get_next_nodes()
        
        # 如果没有下一个节点，结束对话
        if not next_nodes:
            dialogue_prompt = """对话现在应该结束了。
            生成一条礼貌的告别信息来结束对话。信息应当包括：
            1. 感谢用户的花的时间
            2. 祝愿他们一切顺利"""
            
            messages = [
                {"role": "system", "content": dialogue_prompt}
            ]
            generate_params = {
                "do_sample": True,
                "max_new_tokens": 100,
                'repetition_penalty': 1.1,
            }
            
            farewell = call_model(MODEL_URL, messages, generate_params)
            print("Assistant:", farewell)
            break
            
        # 模型A：对话模型生成回复
        dialogue_prompt = f"""#角色:
你现在是加盟商负责招聘骑手的招聘专员，你擅长挖掘对方工作意向，同时你能够引导对方给出他的手机号码，并准确收集并确认对方给出的11位的电话号码。介绍信息时以口语化的形式，不要用书面化的分点的语言，像一个真人那样说话。

Current task: {current_task}
Dialogue history: {dialogue_history}
Generate appropriate response:"""
        
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
        
        # 模型B：用户模型生成回复
        user_prompt = """扮演一个找工作的人,并且自然地回答对方的一切问题，回答尽可能简洁，不要主动说出自己的信息，除非对方问，不要主动展开话题。以口语化的形式回复"""
        messages = [
            {"role": "system", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ]
        
        user_response = call_model(MODEL_URL, messages, generate_params)
        print("User:", user_response)
        dialogue_history.append({"role": "user", "content": user_response})
        
        # 模型C：状态分类器决定下一步
        next_node = state_classifier(user_response, planner.current_node, next_nodes, node_labels)
        
        # 检查是否需要结束对话
        if next_node == 'END_CONVERSATION':
            # 让对话模型生成结束语
            dialogue_prompt = """对话现在应该结束了。
            生成一条礼貌的告别信息来结束对话。信息应当包括：
            1. 感谢用户的花的时间
            2. 祝愿他们一切顺利"""
            
            messages = [
                {"role": "system", "content": dialogue_prompt}
            ]
            generate_params = {
                "do_sample": True,
                "max_new_tokens": 100,
                'repetition_penalty': 1.1,
            }
            
            farewell = call_model(MODEL_URL, messages, generate_params)
            print("Assistant:", farewell)
            break
        
        planner.move_to_node(next_node)

if __name__ == "__main__":
    main() 