from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from rag import rag_app

app=FastAPI()

class QueryRequest(BaseModel):
    question: str
    
class QueryResponse(BaseModel):
    answer: str
    classification: str
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None
    
@app.get("/")

def read_root():
    return {"message": "Süt Sihirbazı API Çalışıyor"}

@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    state = {"question": request.question}
    final_state = rag_app.invoke(state)
    
    response = QueryResponse(
        answer=final_state["answer"],
        classification=final_state["classification"],
        sql_query=final_state.get("query"),
        sql_result=final_state.get("result")
    )
    
    return response

