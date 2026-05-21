import os
import time

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# GEMINI LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# HUGGINGFACE EMBEDDINGS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DATA_PATH = "data/pdfs"
VECTOR_STORE_PATH = "vector_store"

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# =========================
# HUGGINGFACE EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# GEMINI LLM
# =========================

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

# =========================
# INGEST PDFS
# =========================

def ingest_pdfs():

    try:

        loader = PyPDFDirectoryLoader(DATA_PATH)

        documents = loader.load()

        if not documents:
            return {
                "status": "No documents found to ingest"
            }

        print(f"Loaded {len(documents)} pages")

        # BETTER CHUNKING
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks")

        # BATCHING
        batch_size = 32

        print(
            f"Ingesting {len(chunks)} chunks "
            f"in batches of {batch_size}"
        )

        # INITIALIZE VECTORSTORE
        first_batch = chunks[:batch_size]

        vectorstore = FAISS.from_documents(
            first_batch,
            embeddings
        )

        # INGEST REMAINING BATCHES
        for i in range(batch_size, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            print(
                f"Ingesting chunks "
                f"{i} to "
                f"{min(i + batch_size, len(chunks))}"
            )

            vectorstore.add_documents(batch)

            # SMALL DELAY FOR STABILITY
            time.sleep(0.3)

        # SAVE VECTORSTORE
        vectorstore.save_local(VECTOR_STORE_PATH)

        return {
            "status": (
                f"Successfully ingested "
                f"{len(documents)} documents "
                f"({len(chunks)} chunks)"
            )
        }

    except Exception as e:

        import traceback

        print(
            f"Ingestion failed:\n"
            f"{traceback.format_exc()}"
        )

        return {
            "status": "error",
            "message": str(e)
        }

# =========================
# LOAD VECTORSTORE
# =========================

def get_vectorstore():

    try:

        index_file = os.path.join(
            VECTOR_STORE_PATH,
            "index.faiss"
        )

        if not os.path.exists(index_file):
            return None

        return FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    except Exception as e:

        print(f"Vectorstore loading error: {e}")

        return None

# =========================
# QUERY RAG
# =========================

def query_rag(query: str):

    try:

        vectorstore = get_vectorstore()

        if not vectorstore:
            return {
                "response": "No documents ingested yet.",
                "sources": []
            }

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(query)

        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc in docs]
        )

        template = """
You are LawGPT, an expert AI legal assistant.

Answer ONLY using the provided context.

If the answer is not found in the context,
say:
"I could not find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

        prompt = PromptTemplate.from_template(template)

        chain = (
            {
                "context": lambda x: context_text,
                "question": RunnablePassthrough()
            }
            | prompt
            | get_llm()
            | StrOutputParser()
        )

        response = chain.invoke(query)

        sources = [
            {
                "source": doc.metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": doc.metadata.get(
                    "page",
                    "Unknown"
                )
            }
            for doc in docs
        ]

        return {
            "response": response,
            "sources": sources
        }

    except Exception as e:

        return {
            "response": f"RAG Error: {str(e)}",
            "sources": []
        }

# =========================
# SUMMARIZE DOCUMENT
# =========================

def summarize_document(filename: str):

    try:

        vectorstore = get_vectorstore()

        if not vectorstore:
            return "No documents available."

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10}
        )

        docs = retriever.invoke(
            f"Summarize document {filename}"
        )

        if not docs:
            return "No document context found."

        context_text = "\n".join(
            [doc.page_content for doc in docs]
        )

        template = """
Summarize the following legal document.

Context:
{context}

Summary:
"""

        prompt = PromptTemplate.from_template(template)

        chain = (
            prompt
            | get_llm()
            | StrOutputParser()
        )

        return chain.invoke({
            "context": context_text
        })

    except Exception as e:

        return f"Summary Error: {str(e)}"