import os
import sys
import warnings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from openai import OpenAI

warnings.filterwarnings("ignore")

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY or not BASE_URL:
    print("错误：未在 .env 文件中找到 DASHSCOPE_API_KEY 或 BASE_URL")
    sys.exit(1)


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

LLM_MODEL = "qwen-plus" 
DB_DIR = "output/chroma_db"
COLLECTION_NAME = "security_papers"


class SecurityRAGAgent:
    def __init__(self):
        print("\n正在唤醒本地安全知识库与检索神经")
        
        # 加载本地向量库
        if not os.path.exists(DB_DIR):
            print(f"错误：找不到向量数据库路径 {DB_DIR}")
            sys.exit(1)
            
        self.db_client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(anonymized_telemetry=False))
        self.collection = self.db_client.get_collection(name=COLLECTION_NAME)
        
        # 加载 Embedding 模型 (与存入时保持一致)
        self.embed_model = SentenceTransformer("BAAI/bge-m3")
        
        print(f"知识库加载完毕！当前载入网安文献块: {self.collection.count():,} 条\n")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        query_embedding = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
            
        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            retrieved_chunks.append({
                "content": results["documents"][0][i],
                "title": results["metadatas"][0][i].get("paper_title", "Unknown Source"),
                "tags": results["metadatas"][0][i].get("tags", "")
            })
        return retrieved_chunks

    def generate_answer(self, query: str, context_chunks: list):

        if not context_chunks:
            print("\n专家回复: 抱歉，我的安全知识库中未找到与此问题高度相关的文献，无法给出确切解答。")
            return

        context_text = ""
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n【参考资料 {i+1}】(来源: {chunk['title']})\n{chunk['content']}\n"

        # 构建严谨的 System Prompt
        system_prompt = (
            "你是一个顶级的网络安全专家。请【严格基于】下面提供的参考资料来回答用户的问题。\n"
            "要求：\n"
            "1. 逻辑严密，条理清晰（适当使用 Markdown 加粗和列表）。\n"
            "2. 如果参考资料中存在专业术语，请给予解释。\n"
            "3. 绝对不要凭空捏造。如果资料中没有直接答案，请根据已有资料进行合理推断，或明确指出资料的局限性。"
        )
        
        user_prompt = f"【参考资料】:\n{context_text}\n\n【用户问题】: {query}"

        print("\n专家思考中...\n" + "-"*60)
        
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=True  
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n" + "-"*60)
            
            print("\n【本次回答引用来源】:")
            for i, chunk in enumerate(context_chunks):
                print(f"  [{i+1}] {chunk['title'][:80]}... (标签: {chunk['tags']})")
                
        except Exception as e:
            print(f"\n模型请求失败，请检查网络或 API Key: {e}")


def main():
    agent = SecurityRAGAgent()
    print("\n提示: 输入网安问题开始检索，输入 'exit' 或 'q' 退出系统。\n")
    while True:
        try:
            query = input("\n [用户提问]").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'q', 'quit']:
                print("感谢使用，再见！")
                break
            print("正在 10 万篇安全文献中精准检索", end="\r")
            
            chunks = agent.retrieve(query, top_k=3)
            print(" " * 50, end="\r")
            
            agent.generate_answer(query, chunks)
            
        except KeyboardInterrupt:
            print("\n已中断操作。")
            break

if __name__ == "__main__":
    main()