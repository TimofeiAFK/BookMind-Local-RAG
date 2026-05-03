from pathlib import Path
import re

from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from datasets import Dataset
from transformers import AutoTokenizer
from llama_cpp import Llama


# SETTINGS 

BOOK_NAME = input("Write book file name:").strip()
BOOK_PATH = Path(BOOK_NAME)

# WRITE PATH TO MODEL

MODEL_PATH = Path(" ***  ")

OUTPUT_DIR = Path("prepared_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_NAME = "gpt2"

MAX_TOKENS = 512
OVERLAP = 64

N_CTX = 2048
GEN_MAX_TOKENS = 200


# TOKENIZER

def load_tokenizer():
    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# LOAD GGUF MODEL 

def load_llm():
    print("\n--- MODEL CHECK ---")
    print("Model path:", MODEL_PATH)
    print("Model exists:", MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"GGUF model not found:\n{MODEL_PATH}\n"
            "Check MODEL_PATH and exact .gguf filename."
        )

    print("Loading GGUF model...")
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        verbose=False
    )


# EPUB TO TEXT 

def extract_text(epub_path: Path):
    book = epub.read_epub(str(epub_path))
    parts = []

    total_docs = 0
    useful_docs = 0

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            total_docs += 1

            try:
                soup = BeautifulSoup(item.get_content(), "html.parser")

                for tag in soup(["script", "style", "nav"]):
                    tag.decompose()

                text = soup.get_text(separator="\n", strip=True)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()

                if len(text) > 50:
                    parts.append(text)
                    useful_docs += 1

            except Exception as e:
                print(f"Warning: skipped one EPUB section: {e}")

    print(f"EPUB sections found: {total_docs}")
    print(f"Useful sections extracted: {useful_docs}")

    return "\n\n".join(parts)


# CLEAN TEXT 

def clean_text(text: str):
    text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"ISBN[\d\-\sXx:]*", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    text = text.replace("\xa0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    return text.strip()


# CHUNKING 

def chunk_text(text: str, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    print("Total tokens:", len(tokens))

    i = 0
    step = MAX_TOKENS - OVERLAP

    while i < len(tokens):
        chunk_ids = tokens[i:i + MAX_TOKENS]

        if not chunk_ids:
            break

        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        if len(chunk) > 30:
            chunks.append(chunk)

        if i + MAX_TOKENS >= len(tokens):
            break

        i += step

    return chunks


# GENERATION TEST

def generate(llm, text):
    prompt = f"""
You are a helpful assistant.

Use the following book fragment as context and briefly summarize what is happening.

Book fragment:
{text}

Summary:
""".strip()

    output = llm(
        prompt,
        max_tokens=GEN_MAX_TOKENS,
        temperature=0.2,
        top_p=0.9,
        stop=["</s>"]
    )

    print("\n=== MODEL OUTPUT ===")
    print(output["choices"][0]["text"].strip())


# MAIN 

def main():
    print("\n--- FILE CHECK ---")
    print("Current folder:", Path.cwd())
    print("Book path:", BOOK_PATH.resolve())
    print("Book exists:", BOOK_PATH.exists())

    if not BOOK_PATH.exists():
        raise FileNotFoundError(
            f"EPUB file not found:\n{BOOK_PATH.resolve()}\n"
            "Put the EPUB in the same folder as this script or type the full path."
        )

    tokenizer = load_tokenizer()
    llm = load_llm()

    print("\nReading EPUB...")
    raw = extract_text(BOOK_PATH)

    print("Raw length:", len(raw))

    if len(raw.strip()) < 100:
        raise ValueError("EPUB extracted almost nothing. Try another EPUB file.")

    raw_file = OUTPUT_DIR / "raw.txt"
    raw_file.write_text(raw, encoding="utf-8")
    print("Saved:", raw_file)

    print("\nCleaning...")
    text = clean_text(raw)

    clean_file = OUTPUT_DIR / "cleaned.txt"
    clean_file.write_text(text, encoding="utf-8")
    print("Saved:", clean_file)

    print("Clean length:", len(text))

    print("\nChunking...")
    chunks = chunk_text(text, tokenizer)

    print("Chunks:", len(chunks))

    if len(chunks) == 0:
        raise ValueError("No chunks created.")

    chunks_file = OUTPUT_DIR / "chunks.txt"
    chunks_file.write_text("\n\n".join(chunks), encoding="utf-8")
    print("Saved:", chunks_file)

    dataset = Dataset.from_dict({"text": chunks})
    dataset_path = OUTPUT_DIR / "dataset"
    dataset.save_to_disk(str(dataset_path))
    print("Saved dataset:", dataset_path)

    print("\nTesting model...")
    generate(llm, chunks[0][:1000])

    print("\nDONE")


if __name__ == "__main__":
    main()


