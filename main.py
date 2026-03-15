from src.loader import extract_text_from_pdf
from src.cleaner import clean_text, remove_page_noise
from src.chunker import build_chunks
from src.embedder import load_embedding_model, embed_texts
from src.retriever import build_faiss_index, save_artifacts, search_index
from src.llm import load_llm_model, build_rag_prompt, generate_with_llm

# Configuration
PDF_PATH = "./data/raw/document.pdf"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
SAVE_INDEX_PATH = "./vec_database/faiss.index"
SAVE_META_PATH = "./vec_database/chunks_metadata.json"
LLM_NAME = "Qwen3-4B-Instruct-2507-Q6_K.gguf"

TARGET_CHARS = 500
TOP_K = 3

def main():

    # Extract raw page texts from PDF and print total characters
    pages = extract_text_from_pdf(PDF_PATH)
    total_chars = sum(len(p["text"].strip()) for p in pages)
    print(f"Total extracted characters: {total_chars}")
 
    # Remove repeated page headers, then clean and chunk
    pages = remove_page_noise(pages)
    chunks = build_chunks(
        pages,
        target_chars=TARGET_CHARS,
        
    )

    print(f"Built chunks:    {len(chunks)}")
    print("Sample chunk:")
    if chunks:
        print(chunks[13])

    if not chunks:
        print("No chunks were created. Check PDF extraction / cleaning first.")
        return
    
    # Load local embedding model
    model = load_embedding_model(MODEL_NAME)

    # Embed and index
    texts = [c["chunk_text"] for c in chunks]
    vectors = embed_texts(model, texts)
    index = build_faiss_index(vectors)
    print("Embedding dimension:", vectors.shape[1])

    # Save metadata and FAISS index
    save_artifacts(index, chunks, SAVE_INDEX_PATH, SAVE_META_PATH)

    print(f"Saved FAISS index to: {SAVE_INDEX_PATH}")
    print(f"Saved metadata to:    {SAVE_META_PATH}")

    # load llm model
    llm = load_llm_model(LLM_NAME)
    llm_path = llm.model_path

    # Interactive Q&A loop
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        results = search_index(model, index, chunks, query, top_k=TOP_K)

        print("\n=== Retrieved Chunks ===")
        if not results:
            print("No results.")
            continue

        for rank, item in enumerate(results, start=1):
            print(f"\n[{rank}] score={item['score']:.4f} | "
                  f"{item['chunk_id']}")
            print(item["chunk_text"])

        prompt = build_rag_prompt(query, results)
        answer = generate_with_llm(llm_path, prompt)

        print("\n=== Final Answer ===")
        print(answer)


if __name__ == "__main__":
    main()