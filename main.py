import os
import langchain_core
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader

load_dotenv()
api_key = os.getenv("OPENAI_KEY")

st.title("News Research Tool 📈")
st.sidebar.title("News Articles URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

# st.sidebar.button("Process URLs")

processed_url_clicked = st.sidebar.button("Process URLs")

embeddings = OpenAIEmbeddings(api_key=api_key)
main_placeholder = st.empty()
if processed_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    #create embeddings and save it to faiss index
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")

    #save faiss index to directory
    vectorstore.save_local("faiss_index")


llm = llm = OpenAI(api_key = api_key, temperature=0)
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context given bellow.
If the answer is not in the context, say "I don't have enough information to answer this.". Do NOT make things up.

                                          
Context:
{context}

Question:
{question}  

Include sources like 
SOURCES: source urls                              
""")

def docs_sources_retrieval(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

query = main_placeholder.text_input("Question:")
file_path = "./faiss_index"
if query:
    if os.path.exists(file_path):
        vectorIndex = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    retriever = vectorIndex.as_retriever(
        search_kwargs={"k": 4},
        search_type="similarity"
    )

    # Capture docs separately so we can show sources
    retrieved_docs = retriever.invoke(query)
    chain = (
        {
        "context": lambda _: docs_sources_retrieval(retrieved_docs), 
        "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    response = chain.invoke(query)

    st.header("Answer")
    st.write(response)

    #Display sources from retrieved docs
    # st.subheader("Sources:")
    # seen = set()
    # for doc in retrieved_docs:
    #     source = doc.metadata.get("source", "Unknown source")
    #     if source not in seen:
    #         seen.add(source)
    #         st.write(f"- {source}")

