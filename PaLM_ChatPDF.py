import streamlit as st
import os
import base64

# Try to load dotenv, but provide a fallback
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback: manually load .env file if it exists
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import glob


# Sidebar contents
with st.sidebar:
    st.title("🤗💬 LLM Chat App")
    st.markdown(
        """
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Gemini AI](https://makersuite.google.com/app/home) Embeddings & LLM model
 
    """
    )
    add_vertical_space(5)
    st.write("Made with ❤️ by Dji")


def main():
    st.header("Chat with PDF 💬")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())
        files_path = "temp_uploaded.pdf"
    else:
        files_path = "./SOURCE_DOCUMENTS/AIHallucination.pdf"

    loaders = [PyPDFLoader(files_path)]

    # if "index" not in st.session:
    index = VectorstoreIndexCreator(
        embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0),
    ).from_loaders(loaders)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1, convert_system_message_to_human=True)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        return_source_documents=True,
    )

    # st.session.index = index

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")
    # st.write(query)
    # Show PDF Content expander
    with st.expander("Show PDF Content"):
        try:
            with open(files_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.write(f"Could not load PDF: {e}")
    if query:
        response = chain(query)
        st.write(response["result"])        

        # Existing expander for returned chunks
        with st.expander("Returned Chunks"):
            for doc in response["source_documents"]:
                st.write(f"{doc.metadata['source']} \n {doc.page_content}")


if __name__ == "__main__":
    main()
