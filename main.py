from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

def main():
  documents = SimpleDirectoryReader("data").load_data()

  # bge-base embedding model
  Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

  print('hola')
  # ollama
  Settings.llm = Ollama(model="llama3", request_timeout=1.0)

  index = VectorStoreIndex.from_documents(
      documents,
  )

  print(index)

if __name__ == '__main__':
  main()
  