from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def main():
    doc = Path("mhd_doc.txt")
    if not doc.exists():
        doc.write_text("MHD: rho is density, phi is potential.\nLorentz force couples B fields.")

    text = doc.read_text(encoding="utf-8")
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_texts(chunks, embedding=embeddings)

    tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=80)
    llm = HuggingFacePipeline(pipeline=gen)

    query = "What does rho represent in MHD?"
    docs = store.similarity_search(query, k=2)
    context = "\n".join([d.page_content for d in docs])
    prompt = f"Context:\n{context}\n\nQ: {query}\nA:"
    out = llm(prompt)
    print(out)


if __name__ == "__main__":
    main()
