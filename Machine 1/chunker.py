import re

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

from jina_embeddings import JinaCodeEmbeddings


load_dotenv()


# ============================================================
# Load the guideline file
# ============================================================

with open(
    "guidelines.txt",
    "r",
    encoding="utf-8",
) as file:
    text = file.read()


# ============================================================
# Split the file into individual rules
# ============================================================

rules = re.split(
    r"(?=Rule \d+ -)",
    text,
)


# ============================================================
# Create the Jina embedding model
# ============================================================

embeddings = JinaCodeEmbeddings(
    model="jina-embeddings-v2-base-code",
)


# ============================================================
# Prepare examples and metadata
# ============================================================

wrong_examples = []
metadata = []


# Process one rule at a time.
for rule in rules:

    if not rule.strip():
        continue

    # Extract the wrong example.
    wrong_match = re.search(
        r"### Wrong:\s*(.*)",
        rule,
        re.DOTALL,
    )

    if not wrong_match:
        continue

    wrong_example = wrong_match.group(1).strip()

    # This will be embedded.
    wrong_examples.append(wrong_example)

    # This will NOT be embedded.
    # It is only stored as metadata.
    metadata.append(
        {
            "full_rule": rule.strip(),
        }
    )


# ============================================================
# Validate the extracted data
# ============================================================

print(
    f"Number of rules with wrong examples: "
    f"{len(wrong_examples)}"
)

if not wrong_examples:
    raise ValueError(
        "No wrong examples were found in guidelines.txt."
    )


# ============================================================
# Test Jina embedding
# ============================================================

# test_embedding = embeddings.embed_query(
#     wrong_examples[0]
# )

# print(
#     f"Embedding dimension: "
#     f"{len(test_embedding)}"
# )


# ============================================================
# Create FAISS vector database
# ============================================================

vector_db = FAISS.from_texts(
    wrong_examples,
    embeddings,
    metadatas=metadata,
)


# ============================================================
# Save FAISS locally
# ============================================================

vector_db.save_local(
    "faiss_next_index"
)


print(
    "FAISS vector database created successfully."
)


