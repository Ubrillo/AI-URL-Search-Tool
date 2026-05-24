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

llm = OpenAI(api_key = api_key, temperature=0)

def docs_sources_retrieval(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in docs
    )

query = main_placeholder.text_input("Question:")

def stuffing_rag():
    from textwrap import dedent
    prompt = ChatPromptTemplate.from_template(dedent("""
        You are a helpful assistant. Answer the question using only the context provided below.
        If the answer cannot be found in the context, respond with:
        "I don't have enough information to answer this."
        Do not fabricate or infer information beyond what is given.

        Context:
        {context}

        Question:
        {question}

        Format your response as follows:
        Answer: <your answer here>
        Source(s): <list the source url used from the context>
    """).strip())

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
    result = chain.invoke(query)
    return result

def mapping_rag():
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
    # (4) MAP prompt — called once per retrieved chunk
    map_prompt = ChatPromptTemplate.from_template("""
    You are answering a question using a single excerpt from an article.
    Extract any relevant information from the excerpt to help answer the question.
    If the excerpt contains no relevant information, respond with "No relevant info."
    
    Article excerpt (source: {source}):
    {context}

    Question: {question}

    Relevant information:
    """)

    # (5) REDUCE prompt — combines all mapped answers into a final response
    from textwrap import dedent
    synthesis_prompt = ChatPromptTemplate.from_template(dedent("""
        You are a helpful assistant. Your task is to produce a single, coherent answer
        to the question below by combining information from multiple article excerpts.

        Guidelines:
        - Use only the information provided in the excerpts. Do not fabricate or infer beyond it.
        - Avoid repeating the same point more than once, even if multiple excerpts mention it.
        - If excerpts contradict each other, acknowledge the conflict briefly and present both views.
        - If none of the excerpts contain a satisfactory answer, say: "I don't have enough information to answer this."

        Excerpts and their sources:
        {mapped_answers}

        Question: {question}

        Format your response as follows:
        Answer: <your synthesized answer here>
        Source(s): <list the source title(s) or url you drew from>.
    """).strip())

    def map_docs(inputs: dict) -> dict:
        question = inputs["question"]
        retrieved_docs = retriever.invoke(question)
        
        mapped = []
        for doc in retrieved_docs:
            result = map_chain.invoke({
                "context": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "question": question
            })
            mapped.append(f"[Source: {doc.metadata.get('source', 'unknown')}]\n{result}")
        
        return {
            "mapped_answers": "\n\n---\n\n".join(mapped),
            "question": question
        }
    
    # (6) Map step — run LLM individually on each retrieved chunk
    map_chain = map_prompt | llm | StrOutputParser()

    # (7) Reduce step — synthesize mapped answers into final response
    reduce_chain = synthesis_prompt | llm | StrOutputParser()

    # (8) Full map-reduce chain
    map_reduce_chain = RunnableLambda(map_docs) | RunnableLambda(
        lambda x: reduce_chain.invoke(x)
    )

    result = map_reduce_chain.invoke({"question": query})
    return result

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

    response = mapping_rag()

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