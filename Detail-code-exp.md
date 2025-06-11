## Code flow explanation: Chat with PDF

<<Diagram>>

### Overview
This Streamlit app allows users to upload a PDF and then ask questions about its content. 

The app uses:
* LangChain for chaining together LLMs and retrieval.
* Google Gemini AI for both embeddings and chat LLM.
* Vector Store to store and search PDF content efficiently.
* RetrievalQA (RAG) to combine LLM with document retrieval.

### Step-by-step flow
#### PDF Upload & Loading
* User uploads a PDF (or a default PDF is used).
* The PDF is loaded using `PyPDFLoader`.

#### Text Splitting & Embedding
* The PDF is split into chunks using `RecursiveCharacterTextSplitter`.
* Each chunk is converted into a vector (embedding) using `GoogleGenerativeAIEmbeddings`.

#### Vector Store Creation
* All chunk embeddings are stored in a vector store (database for vectors).
* This allows for efficient similarity search.

#### Retrieval-Augmented Generation (RAG)
When a user asks a question, the app:
1. Converts the question into an embedding.
2. Searches the vector store for the most relevant PDF chunks.
3. Passes those chunks, along with the question, to the LLM (Gemini) to generate an answer.

#### Display
* The answer and the relevant PDF chunks are displayed to the user.

#### Key Concepts
* LangChain: A framework to build applications with LLMs and external data (like PDFs).
* RAG (Retrieval-Augmented Generation): Combines retrieval (searching for relevant info) with generation (LLM answers).
* Vector Store: A database that stores vector representations of text for fast similarity search.

##### MermaidJS Flowchart script
```
flowchart TD
    A["User uploads PDF or uses default"] --> B["Load PDF with PyPDFLoader"]
    B --> C["Split PDF into chunks (RecursiveCharacterTextSplitter)"]
    C --> D["Convert chunks to embeddings (GoogleGenerativeAIEmbeddings)"]
    D --> E["Store embeddings in Vector Store"]
    E --> F["User asks a question"]
    F --> G["Convert question to embedding"]
    G --> H["Search Vector Store for relevant chunks"]
    H --> I["Send question + chunks to LLM (Gemini)"]
    I --> J["LLM generates answer"]
    J --> K["Display answer and relevant chunks to user"]
```