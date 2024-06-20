vector_store = Chroma(persist_directory="/myfolder/chroma_db")

for collection in vector_store._client.list_collections():
  ids = collection.get()['ids']
  print('REMOVE %s document(s) from %s collection' % (str(len(ids)), collection.name))
  if len(ids): collection.delete(ids)