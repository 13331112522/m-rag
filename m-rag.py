import imageio
import os
import argparse
import ollama
import glob
from tqdm import tqdm
import time
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.embeddings import GPT4AllEmbeddings,HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import GPT4All, LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
load_dotenv()

# Load environment variables
source_directory = os.environ.get('SOURCE_DIRECTORY')
stride= os.environ.get('STRIDE')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
# Replace with the actual path to your FFmpeg executable
os.environ["IMAGEIO_FFMPEG_EXE"] = "/path/to/ffmpeg"

def img_parse(img_path):
    res = ollama.chat(
	            model="llava",
	            messages=[
		            {
			            'role': 'user',
			            'content': 'Describe this image:',
			            'images': [img_path]
		            }
	            ]
            )

    with open(source_directory+"/"+os.path.basename(img_path)+".txt", "a") as write_file:
        write_file.write("---"*10 + "\n\n")
        write_file.write(os.path.basename(img_path) + "\n\n")
        write_file.write(res['message']['content'])
        write_file.flush()
    print("Proceeding "+img_path)
#parse the video with llava into txt if source_documents is empty
def video_parse(video_path):
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    try:
        total_frames = meta["n_frames"]  # Access frame count from metadata (if available)
    except KeyError:
        print("Frame count not found in metadata. Counting frames manually...")
        total_frames = 0
        for _ in reader:
            total_frames += 1

    print(f"Number of frames: {total_frames}")

    for i in tqdm(range(total_frames)):

        frame = reader.get_next_data()

        if i % int(stride) == 0: 
            # Save the image to a file
            imageio.imsave('temp.png', frame)
            res = ollama.chat(
	            model="llava",
	            messages=[
		            {
			            'role': 'user',
			            'content': 'Describe this image:',
			            'images': ['./temp.png']
		            }
	            ]
            )

            with open(source_directory+"/"+os.path.basename(video_path)+".txt", "a") as write_file:
                write_file.write("---"*10 + "\n\n")
                write_file.write(os.path.basename(video_path)+"(Frame:"+str(i) +")" + "\n\n")
                write_file.write(res['message']['content'])
                write_file.flush()
        
    reader.close()
    print("Proceeding "+video_path)

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc
#Custom video loader
class VideoLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> List[Document]:
        try:
            video_parse(self.file_path)
            loader=TextLoader(source_directory+"/"+os.path.basename(self.file_path)+".txt")
            doc=loader.load()
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e
        return doc

class ImgLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self) -> List[Document]:
        try:
            img_parse(self.file_path)
            loader=TextLoader(source_directory+"/"+os.path.basename(self.file_path)+".txt")
            doc=loader.load()
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e
        return doc

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".mp4": (VideoLoader,{}),
    ".jpg": (ImgLoader,{}),
    ".png": (ImgLoader,{}),
    # Add more mappings for other file extensions and loaders as needed
}

def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]
    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            
            for j, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                
                results.extend(docs)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts

# The mail loop
def main():
    #embeddings = GPT4AllEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    try:
        os.listdir("faiss_index")
  
    except FileNotFoundError:
        # Create and store locally vectorstore if folder not exit
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local("faiss_index")
        print(f"Ingestion complete! You can now query your visual documents")

    #loading the vectorstore
    db=FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
   
   # Callbacks support token-wise streaming
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_gpu_layers=1, n_batch=512, f16_kv=True,callback_manager=callback_manager,verbose=True)   
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callback_manager=callback_manager,verbose=True)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
        
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue
        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if False else res['source_documents']
        end = time.time()
        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)
        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

if __name__ == "__main__":
    main()
