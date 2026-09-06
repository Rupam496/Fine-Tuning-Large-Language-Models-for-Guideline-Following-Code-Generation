import os

from dotenv import load_dotenv
from jina_embeddings import JinaCodeEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# Hyperparameters
# ============================================================

TOP_K = 3           # Number of guidelines retrieved



# ============================================================
# Paths
# ============================================================

FAISS_PATH = "/home/rupamdas/Rupam/ET project/Code generation following guidelines/faiss_next_index"


# ============================================================
# Load API key
# ============================================================

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")




# ============================================================
# Load Voyage embedding model
# ============================================================

embeddings = JinaCodeEmbeddings(
    model="jina-embeddings-v2-base-code",
)


# ============================================================
# Load FAISS vector database
# ============================================================

vector_db = FAISS.load_local(
    FAISS_PATH,
    embeddings,
    allow_dangerous_deserialization=True,
)



def get_judge_prompt(question, codes):

    retrieval_prompt = f"""
    Question:
    {question}
    
    """
    
    for i, code in enumerate(codes, start=1):

        retrieval_prompt += f"""
Code {i}:
{code}

"""

    retrieved_guidelines = vector_db.similarity_search(
            retrieval_prompt,
            k=TOP_K
        )

    judge_prompt = f"""
    You are a code evaluation judge.
    
    Evaluate the generated Python codes according to the
    provided coding guidelines.
    
    Question:
    {question}
    
"""

    for i, code in enumerate(codes, start=1):

        judge_prompt += f"""
Code {i}:
{code}

"""

    judge_prompt += """
Retrieved Guidelines:
"""

    for i, document in enumerate(retrieved_guidelines, start=1):

        judge_prompt += f"""
Guideline {i}:
{document.metadata["full_rule"]}

"""
    return judge_prompt