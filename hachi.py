from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
import gradio as gr
import uuid


llm = Ollama(model="llama3")
loader = TextLoader("./data.txt")
docs = loader.load()
embeddings = OllamaEmbeddings(model="llama3")
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, llm=llm
)

# Contextualize question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, vector.as_retriever(), contextualize_q_prompt
)

# Answer question
qa_system_prompt = """
    You are a helpful AI bot. Your name is Hachiko.
    You are supposed to help people answer questions about Kaashvi as they are playing a trivia game.
    Don't give them out direct answers, but help them find the answers.
    You can give them specific hints. Don't give them hints about Kaashvi for questions which are unrelated to the context.
    Since most of the people playing the trivia are
    from Australia, hints should be relevant to Australia. Also, keep the responses brief up to one paragraph.
    Hachiko happens to be a pet dog's name, hence it would be funny if you could make some dog-related jokes as a part of the hint.
    
    {context}               
    """

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Below we use create_stuff_documents_chain to feed all retrieved context # into the LLM. Note that we can also use StuffDocumentsChain and other # instances of BaseCombineDocumentsChain.
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
formatted_chat_history = []

with gr.Blocks() as chat_system:
    gr.State(value=uuid.uuid4())
    initial_message = AIMessage(content="Hi! My name is Hachiko. Woof Woof")
    chat = gr.Chatbot(
        value=[[None, initial_message.content]],
        placeholder="Chat with Hachiko",
    )
    prompt = gr.Textbox(placeholder="Ask me anything about Kaashvi")
    submit = gr.Button(value="Submit", variant="primary")
    clear = gr.ClearButton([prompt, chat])

    def llm_reply(question, chat_history):
        formatted_chat_history.append(HumanMessage(content=question))
        reply = rag_chain.invoke(
            {"input": question, "chat_history": formatted_chat_history}
        )["answer"]
        formatted_chat_history.append(AIMessage(content=reply))
        chat_history.append((question, reply))
        return "", chat_history

    submit.click(llm_reply, [prompt, chat], [prompt, chat])
    prompt.submit(llm_reply, [prompt, chat], [prompt, chat])

chat_system.launch(share=True)
