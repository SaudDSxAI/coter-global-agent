import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# ================= CONFIG =================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path("data")
COMBINED_FILE = DATA_DIR / "combined.txt"
FAISS_PATH = DATA_DIR / "faiss_index"
PROMPT_FILE = Path("prompt.txt")  # still from main directory

if not OPENAI_API_KEY:
    raise ValueError("❌ Missing OPENAI_API_KEY in .env")

os.makedirs(DATA_DIR, exist_ok=True)

# ================= STEP 0: Load Prompt =================
def load_prompt():
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"❌ Missing prompt file: {PROMPT_FILE}")
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    return SystemMessagePromptTemplate.from_template(
        f"You are Saud's AI Assistant.\nAlways follow these recruiter-oriented instructions:\n\n{text}"
    )

# ================= STEP 1: Load Data Files =================
def load_all_files(data_dir=DATA_DIR):
    print(f"📂 Scanning folder: {data_dir}")
    texts = []

    for file in data_dir.glob("*"):
        if file.suffix.lower() == ".txt":
            loader = TextLoader(str(file), encoding="utf-8")
        elif file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
        elif file.suffix.lower() in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(str(file))
        else:
            print(f"⚠️ Skipping unsupported file: {file.name}")
            continue

        try:
            docs = loader.load()
            texts.extend([doc.page_content for doc in docs])
            print(f"✅ Loaded: {file.name}")
        except Exception as e:
            print(f"⚠️ Failed to load {file.name}: {e}")

    if not texts:
        raise ValueError("❌ No valid files found in data folder.")
    return texts

# ================= STEP 2: Combine and Save =================
def combine_and_save(texts, output_file=COMBINED_FILE):
    combined_text = "\n\n".join(texts)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(combined_text)
    print(f"✅ Combined text saved → {output_file}")
    return combined_text

# ================= STEP 3: Create or Load FAISS =================
def get_vectorstore(text, faiss_path=FAISS_PATH):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")

    if faiss_path.exists():
        try:
            print("📦 Found FAISS index, loading...")
            return FAISS.load_local(str(faiss_path), embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"⚠️ Failed to load FAISS: {e} — rebuilding.")

    print("🧠 Creating new FAISS index...")
    vectorstore = FAISS.from_texts([text], embeddings)
    vectorstore.save_local(str(faiss_path))
    print(f"✅ FAISS index saved → {faiss_path}")
    return vectorstore

# ================= STEP 4: Build QA Chain =================
def build_qa_chain(vectorstore, system_prompt):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    human = HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
    prompt = ChatPromptTemplate.from_messages([system_prompt, human])

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# ================= MAIN PIPELINE =================
def run_pipeline(init_only: bool = False):
    print("🔄 Loading recruiter prompt...")
    system_prompt = load_prompt()

    print("📥 Loading data files...")
    texts = load_all_files()

    print("🧩 Combining text...")
    combined_text = combine_and_save(texts)

    print("🧠 Building or loading embeddings...")
    vectorstore = get_vectorstore(combined_text)

    print("⚙️ Building QA chain...")
    qa_chain = build_qa_chain(vectorstore, system_prompt)

    if init_only:
        return qa_chain

    print("\n🚀 Assistant ready! Type 'exit' to quit.\n")
    while True:
        query = input("Query: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        try:
            result = qa_chain.invoke({"query": query})
            print("\n💬 Answer:", result["result"])
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    run_pipeline()
