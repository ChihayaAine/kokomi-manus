import os
import json
import time
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from openai import OpenAI

# 设置 OpenAI 客户端
client = OpenAI(
    api_key="",
    base_url=""
)
# 定义函数：从文本文件中提取文本
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()

# 定义函数：将文本分割成块
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# 定义函数：生成问题和答案 - 修改后不再指定问题数量
def generate_qa_pairs(chunk):
    prompt = f"""
基于以下给定的文本，生成一组高质量的问答对。请遵循以下指南：
        
                1. 问题部分：
                - 为同一个主题创建尽可能多的（如K个）不同表述的问题，确保问题的多样性。
                - 每个问题应考虑用户可能的多种问法，例如：
                - 直接询问（如"什么是...？"）
                - 请求确认（如"是否可以说...？"）
                - 寻求解释（如"请解释一下...的含义。"）
                - 假设性问题（如"如果...会怎样？"）
                - 例子请求（如"能否举个例子说明...？"）
                - 问题应涵盖文本中的关键信息、主要概念和细节，确保不遗漏重要内容。

                2. 答案部分：
                - 提供一个全面、信息丰富的答案，涵盖问题的所有可能角度，确保逻辑连贯。
                - 答案应直接基于给定文本，确保准确性和一致性。
                - 包含相关的细节，如日期、名称、职位等具体信息，必要时提供背景信息以增强理解。

                3. 格式：
                - 使用json格式返回,里面包含多个问答对
                - 使用 "question:" 标记问题的开始。
                - 使用 "answer:" 标记答案的开始，答案应清晰分段，便于阅读。
                - 问答对之间用两个空行分隔，以提高可读性。

                4. 内容要求：
                - 确保问答对紧密围绕文本主题，避免偏离主题。
                - 避免添加文本中未提及的信息，确保信息的真实性。
                - 如果文本信息不足以回答某个方面，可以在答案中说明 "根据给定信息无法确定"，并尽量提供相关的上下文。

                5. 示例结构（仅供参考，实际内容应基于给定文本）：
                [
                {{"question":"问题1", "answer":"答案1"}},
                {{"question":"问题2", "answer":"答案2"}},
                {{"question":"问题3", "answer":"答案3"}},
                {{"question":"问题4", "answer":"答案4"}},
                {{"question":"问题5", "answer":"答案5"}},
                {{"question":"问题6", "answer":"答案6"}}
                ]
            给定文本：
            {chunk}

            请基于这个文本生成问答对。
            """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.1,
            max_tokens=8000
        )
        
        content = response.choices[0].message.content.strip()
        
        # 尝试解析JSON
        try:
            qa_pairs = json.loads(content)
            return qa_pairs
        except json.JSONDecodeError:
            # 如果无法解析JSON，尝试从文本中提取JSON部分
            match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
            if match:
                try:
                    qa_pairs = json.loads(match.group(0))
                    return qa_pairs
                except:
                    pass
            
            # 如果仍然失败，尝试解析Q:/A:格式
            try:
                qa_pairs = []
                parts = content.split("\n\n\n")
                for part in parts:
                    if "Q:" in part and "A:" in part:
                        qa_part = part.split("A:", 1)
                        question = qa_part[0].replace("Q:", "").strip()
                        answer = qa_part[1].strip()
                        qa_pairs.append({"question": question, "answer": answer})
                if qa_pairs:
                    return qa_pairs
            except:
                pass
            
            # 如果仍然失败，返回空列表
            print(f"Failed to parse response as JSON: {content}")
            return []

    except Exception as e:
        print(f"Error generating QA pairs: {str(e)}")
        return []

# 定义函数：保存问答对到本地文件
def save_qa_pairs_locally(qa_pairs, filename, format="json"):
    # 创建输出目录（如果不存在）
    output_dir = "qa_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 构建完整的文件路径
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_path = os.path.join(output_dir, f"{base_filename}_qa_pairs.{format}")
    
    # 保存文件
    if format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    elif format == "csv":
        df = pd.DataFrame(qa_pairs)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    return output_path

# 主函数：处理文件并生成QA对
def process_file(file_path, chunk_size=1000, chunk_overlap=100):
    print(f"处理文件: {file_path}")
    
    # 提取文本
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.txt'):
        text = extract_text_from_txt(file_path)
    else:
        print(f"不支持的文件类型: {file_path}")
        return
        
    print(f"文本长度: {len(text)} 字符")
    
    # 分割文本
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    print(f"分割为 {len(chunks)} 个文本块")
    
    # 生成问答对
    all_qa_pairs = []
    
    for i, chunk in enumerate(chunks):
        print(f"处理文本块 {i+1}/{len(chunks)}...")
        
        qa_pairs = generate_qa_pairs(chunk)
        all_qa_pairs.extend(qa_pairs)
        
        # 避免API限制
        if i < len(chunks) - 1:
            time.sleep(1)
    
    print(f"生成了 {len(all_qa_pairs)} 个问答对")
    
    # 保存到本地
    json_path = save_qa_pairs_locally(all_qa_pairs, file_path, "json")
    csv_path = save_qa_pairs_locally(all_qa_pairs, file_path, "csv")
    
    print(f"JSON 文件已保存至: {json_path}")
    print(f"CSV 文件已保存至: {csv_path}")
    
    return all_qa_pairs

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='从文件生成问答对')
    parser.add_argument('--file_path', '-f', 
                        default='/Users/weilei/Desktop/llmeva/RAG-QA-Generator/Data/testdata.txt', 
                        help='要处理的文件路径 (TXT)')
    parser.add_argument('--chunk-size', '-c', type=int, default=1000, help='文本块大小')
    parser.add_argument('--chunk-overlap', '-o', type=int, default=100, help='文本块重叠')
    
    args = parser.parse_args()
    
    process_file(args.file_path, args.chunk_size, args.chunk_overlap)