from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama


# SETTINGS 

CHUNKS_FILE = Path("prepared_data/chunks.txt")
# WRITE PATH TO MODEL
MODEL_PATH = Path(" *** ")

EMBED_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

TOP_K = 12
N_CTX = 8192
MAX_NEW_TOKENS = 350


# LOAD CHUNKS 

def load_chunks(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks file not found:\n{path.resolve()}\n"
            "Run preprocessing first."
        )

    text = path.read_text(encoding="utf-8")
    raw_chunks = re.split(r"\n{2,}", text)

    bad_words = [
        "table of contents",
        "contents",
        "glossary",
        "appendix",
        "notes",
        "index",
        "copyright",
        "publisher",
        "isbn"
    ]

    chunks = []

    for chunk in raw_chunks:
        chunk = chunk.strip()

        if len(chunk) < 120:
            continue

        low = chunk.lower()

        # remove technical/TOC chunks
        if any(bad in low for bad in bad_words):
            continue

        chunks.append(chunk)

    if not chunks:
        raise ValueError("No usable chunks found after filtering.")

    print(f"Loaded chunks: {len(chunks)}")
    print("\nFirst chunk preview:")
    print(chunks[0][:500])
    print("-" * 80)

    return chunks


# EMBEDDINGS 

def load_embedder():
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    return SentenceTransformer(EMBED_MODEL_NAME)


def build_index(chunks, embedder):
    print("Encoding chunks...")
    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embeddings


# LOAD LOCAL MODEL 

def load_llm():
    print("\nChecking GGUF model path...")
    print("Model path:", MODEL_PATH.resolve())
    print("Exists:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GGUF model not found:\n{MODEL_PATH.resolve()}\n"
            "Put model into models/ and rename it to mistral.gguf"
        )

    print("Loading GGUF model...")
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        verbose=False
    )


# RETRIEVAL 

def retrieve(query, chunks, embeddings, embedder, top_k=8):
    q = query.lower()

    # If question asks about beginning, take first chunks
    beginning_keywords = [
        "beginning",
        "start",
        "opening",
        "first chapter",
        "first chapters",
        "first part",
        "summary",
        "summarize",
        "overview",
        "plot"
    ]

    if any(keyword in q for keyword in beginning_keywords):
        return [(chunk, 1.0) for chunk in chunks[:top_k]]

    # Normal semantic search
    query_emb = embedder.encode([query], convert_to_numpy=True)
    scores = cosine_similarity(query_emb, embeddings)[0]

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]


# ANSWER 

def answer_question(llm, question, retrieved_chunks):
    context = "\n\n".join(
        [
            f"[Fragment {i + 1}]\n{chunk}"
            for i, (chunk, _) in enumerate(retrieved_chunks)
        ]
    )

    prompt = f"""
You are a strict book QA assistant.

Use ONLY the book fragments below.
Answer the user's question directly.
Do NOT create new questions.
Do NOT continue with extra Q&A examples.
Do NOT invent facts.
Do NOT use knowledge from outside the fragments.

If the fragments contain partial evidence, answer using that evidence. Only say not found if the fragments are completely unrelated.

Book fragments:
{context}

User question:
{question}

Final answer:
""".strip()

    output = llm(
        prompt,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=0.8,
        repeat_penalty=1.15,
        stop=[
            "User question:",
            "Question:",
            "\nQuestion:",
            "\n\nQuestion:",
            "Book fragments:",
            "Final answer:",
            "</s>"
        ]
    )

    answer = output["choices"][0]["text"].strip()

    # Safety cut if model starts making new Q/A
    cut_markers = [
        "\nQuestion:",
        "\nUser question:",
        "\nBook fragments:",
        "\nFinal answer:",
        "\nAnswer:"
    ]

    for marker in cut_markers:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    return answer


# MAIN 

def main():
    chunks = load_chunks(CHUNKS_FILE)

    embedder = load_embedder()
    embeddings = build_index(chunks, embedder)

    llm = load_llm()

    print("\nBook RAG is ready.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Good bye.")
            break

        if not question:
            continue

        retrieved = retrieve(
            question,
            chunks,
            embeddings,
            embedder,
            top_k=TOP_K
        )

        print("\nTop fragments:")
        for i, (chunk, score) in enumerate(retrieved, start=1):
            print(f"\n--- Fragment {i}, similarity={score:.4f} ---")
            print(chunk[:700])

        answer = answer_question(llm, question, retrieved)

        print("\nAssistant:")
        print(answer)
        print("\n" + "-" * 80 + "\n")


if __name__ == "__main__":
    main()

