import chromadb
from chromadb.utils import embedding_functions

# Используем локальную модель для эмбеддингов
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Создаем локальную базу данных
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="it_faq", embedding_function=default_ef)

# Читаем файл
with open("knowledge_base.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Разбиваем текст по слову "Вопрос:"
chunks = []
parts = text.split("Вопрос:")
for part in parts:
    if part.strip():
        chunks.append("Вопрос:" + part.strip())

# Очищаем базу, если там уже что-то было
if collection.count() > 0:
    collection.delete(ids=collection.get()['ids'])

# Добавляем в базу
collection.add(
    documents=chunks,
    ids=[f"doc_{i}" for i in range(len(chunks))]
)
print(f"Успешно добавлено {len(chunks)} вопросов в базу знаний.")