from llama_cpp import Llama
from typing import List, Dict
import os

def load_llm_model(llm_name: str) -> Llama:

    model_path = "./models"
    os.makedirs(model_path, exist_ok=True)

    llm = Llama.from_pretrained(
	repo_id="unsloth/Qwen3-4B-Instruct-2507-GGUF",
	filename=llm_name,
    local_dir=model_path,
    verbose=False
    )
    return llm

def build_rag_prompt(question: str, retrieved_chunks: List[Dict], max_tokens: int = 3000) -> str:

    # sort chunks by score
    retrieved_chunks = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)

    context_blocks = []
    current_tokens = 0

    for rank, chunk in enumerate(retrieved_chunks, start=1):

        block = (
            f"[Rank {rank} | {chunk['chunk_id']} | "
            f"score {chunk['score']:.4f}]\n"
            f"{chunk['chunk_text']}"
        )

        # one token roughly 1.5 to 1.8 characters for Chinese
        block_tokens = len(block) // 1.5

        # stop if exceeding limit
        if current_tokens + block_tokens > max_tokens:
            break

        context_blocks.append(block)
        current_tokens += block_tokens

    context_text = "\n\n".join(context_blocks)
    
    prompt = f""" Please help answer the question based on the provided context. 
    If you find an answer, please point out the supporting chunk's id and all their contents as they are presented and then give a final answer.
    If the answer is not contained in the context or the context is not enough for answering the question, say: "無法回覆該問題".
    
    Related Context: {context_text}
    Question: {question}
    """
    return prompt

def generate_with_llm(llm_path: str, prompt: str) -> str:
    
    llm = Llama(model_path=llm_path, n_ctx=4096,verbose=False)
    
    system_prompt = """You are a helpful assistant for document question answering. 
    You will provided with one or more questions and question-related chunks of a legal document. 
    The ansers to different questions could lies in different chunks, and some questions may require multiple chunks to answer. 
    Please carefully read the content of each chunk and give a comprehensive answer to the questions.
    Use only the provided context to answer the questions. Do not make up any information that is not in the context.
    The document is named 性騷擾防治法 and the revised date is 民國 112 年 08 月 16 日. 
    The document is in traditional Chinese. Please answer in traditional Chinese as well.
    """

    output = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=4096
    )
    return output["choices"][0]["message"]["content"].strip()