from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_function import *


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8501",  # Streamlit 기본 포트
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def similarity_search(query: QueryRequest):
    question = query.query
    
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    results = getStreamingChain(question)
        
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=819)