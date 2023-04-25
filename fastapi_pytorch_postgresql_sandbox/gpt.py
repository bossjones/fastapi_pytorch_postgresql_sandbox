"""fastapi_pytorch_postgresql_sandbox.main"""
# sourcery skip: avoid-global-variables
# pylint: disable=no-name-in-module
# pylint: disable=no-member
# SOURCE: https://levelup.gitconnected.com/how-to-create-a-doc-chatbot-that-learns-everything-for-you-in-15-minutes-364fef481307
import os

from langchain import OpenAI
from llama_index import (
    GPTSimpleVectorIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    download_loader,
)
import streamlit as st

from fastapi_pytorch_postgresql_sandbox.settings import settings

os.environ["OPENAI_API_KEY"] = settings.openai_token
DOC_PATH = "./gpt_data/"
INDEX_FILE = "index.json"

if "response" not in st.session_state:
    st.session_state.response = ""

index = None
st.title("Yeyu's Doc Chatbot")


def send_click() -> None:
    """_summary_"""
    st.session_state.response = index.query(st.session_state.prompt)


sidebar_placeholder = st.sidebar.container()
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    doc_files = os.listdir(DOC_PATH)
    for doc_file in doc_files:
        os.remove(DOC_PATH + doc_file)

    bytes_data = uploaded_file.read()
    with open(f"{DOC_PATH}{uploaded_file.name}", "wb") as f:
        f.write(bytes_data)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    loader = SimpleDirectoryReader(DOC_PATH, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    sidebar_placeholder.header("Current Processing Document:")
    sidebar_placeholder.subheader(uploaded_file.name)
    sidebar_placeholder.write(f"{documents[0].get_text()[:10000]}...")

    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, model_name="text-davinci-003"),
    )

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
    )

    index = GPTSimpleVectorIndex.from_documents(
        documents,
        service_context=service_context,
    )

    index.save_to_disk(INDEX_FILE)

elif os.path.exists(INDEX_FILE):
    index = GPTSimpleVectorIndex.load_from_disk(INDEX_FILE)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(DOC_PATH, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    doc_filename = os.listdir(DOC_PATH)[0]
    sidebar_placeholder.header("Current Processing Document:")
    sidebar_placeholder.subheader(doc_filename)
    sidebar_placeholder.write(f"{documents[0].get_text()[:10000]}...")

if index != None:
    st.text_input("Ask something: ", key="prompt")
    st.button("Send", on_click=send_click)
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon="🤖")
