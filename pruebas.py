from langchain_community.document_loaders import Docx2txtLoader #To load word files
from langchain.document_loaders import PyPDFLoader #To load pdf files
from langchain.vectorstores import Chroma #Vectorial DataBase
from langchain.embeddings.openai import OpenAIEmbeddings #Word2Vec Model to tokenizer the text
from langchain.text_splitter import RecursiveCharacterTextSplitter #To splitt the text
from langchain.chat_models import ChatOpenAI #To load LLM from OPENAI
from langchain.chains import RetrievalQA #To build chains to QA tasks
from langchain import PromptTemplate #Class that allow the before to get in in production
from langchain.chains.summarize import load_summarize_chain #To buil chains to summarize tasks
import os
from langchain.vectorstores import Chroma #The vectorial database
from langchain_community.document_loaders import TextLoader




def getFiles(route):
    #CARRY THE TXT FILES CONTENT TO LANGCHAIN FORMAT:
    concatenated_files=[]
    root_path=route +'/'
    txt_files=list(os.listdir(route))
    for txt in txt_files:
        loader = TextLoader(root_path+txt)
        data=loader.load()
        concatenated_files.extend(data)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, #Fragments of text of 1500 tokens
        chunk_overlap=200, #For evey fragment that take the 200 last tokens of the last fragment
        length_function=len
        )

    documents = text_splitter.split_documents(concatenated_files) #List with the metadata and the content splitt by fragments of 1500 tokens

    return documents 

    #As we can see, the LLM can procces a limit amount of tokens, so that we have to split the text in fragments of 1500 tokens in this case (because is the maximun amount of tokens that support our model)


def configureOPENAI():
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") #word2vec model of openAI
    return embeddings


def generateDatabase(name_database,embedding_model,documents):
    #Creating our vectorial database or vector store
    vectorstore_chroma = Chroma.from_documents(
        documents=documents, #Create the database with the list of the created documents (Every instance will be the embedding of every document)
        embedding=embedding_model, #Word2vec model to create our embeddings, always use the same.
        persist_directory=name_database #Load my database in the indicated folder (If I close the section, I will keep storaged my vectorial databas in the folder called "NOMBRE_INDICE_CHROMA" )
    )

    return vectorstore_chroma


def testing():

    ## CARGAMOS LOS ARCHIVOS
    
    documents = getFiles('FilesData')

    ## CONFIGURAMOS EL MODELO Y LA API DE OPENAI

    embedding_model = configureOPENAI()

    ## GENERAMOS LA VECTORIZACIÓN DE LA BASE DE DATOS

    vector_chroma = generateDatabase('Testing_database',embedding_model,documents)

    ## HACEMOS LA PREGUNTA
    query = "¿En que fecha fue la revolución francesa?"
    docs = vector_chroma.similarity_search(query,k=3)
    print("DOCUMENTOS CERCANOS: ",docs)


if __name__ == "__main__":

    testing()