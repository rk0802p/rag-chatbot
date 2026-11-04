# Legacy FAISS Store (Archived)

These files (`index.faiss` and `index.pkl`) are legacy artifacts created in initial commit `eb52e5e` using LangChain's `InMemoryDocstore` and `faiss.FAISS.save_local("faiss_db")`.

### Compatibility Note:
- The current production RAG architecture avoids Python `pickle` deserialization for security reasons (mitigating arbitrary code execution risks) and avoids LangChain coupling.
- The active application uses native `faiss.write_index()` / `faiss.read_index()` alongside structured JSON metadata.
- These files are retained solely for repository version history and are not loaded by the active pipeline.
