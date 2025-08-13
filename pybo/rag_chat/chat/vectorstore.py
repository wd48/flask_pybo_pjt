import chromadb
from config import CHAT_DB_PERSIST_DIR

# ChromaDB에서 컬렉션 목록을 가져오는 함수
def get_collection_list():
    try:
        persistent_client = chromadb.PersistentClient(CHAT_DB_PERSIST_DIR)
        collections = persistent_client.list_collections()
        print(f"[-2XY-] Available collections: {len(collections)}")
        return collections
    except Exception as e:
        print(f"[-2XY-] Error retrieving collections: {e}")
        return None

# ChromaDB에서 컬렉션 이름을 가져오는 함수
def get_collection_names():
    collection_names = []
    collections = get_collection_list()

    if collections is None:
        print("[-2XY-] No collections found")
        return collection_names

    for collection in collections:
        try:
            print(f"[-2XY-] Collection name: {collection.name}")
            collection_name = collection.name
            collection_names.append(collection.name)
        except AttributeError as e:
            print(f"[-2XY-] Error accessing collection name: {e}")

    return collection_names
