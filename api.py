import os
import io
from typing import Optional, Literal
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, END, StateGraph
from typing_extensions import TypedDict

# --- YAPILANDIRMA VE ORTAM DEĞİŞKENLERİ ---
load_dotenv()

# --- VERİ TİPLERİ (Pydantic Models) ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    classification: str
    answer: str
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None

class State(TypedDict):
    question: str
    classification: str 
    query: str
    result: str
    answer: str

# --- BAĞIMLILIKLAR VE GLOBAL DEĞİŞKENLER ---
db: Optional[SQLDatabase] = None
llm: Optional[ChatOllama] = None
app_workflow = None

def get_database():
    try:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        # Eğer port .env'de yoksa varsayılan 5432 kullanılır
        db_port = os.getenv("DB_PORT", "5432")
        
        db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        print(f"Veritabanı bağlantı hatası: {e}")
        return None

# --- LANGGRAPH DÜĞÜM FONKSİYONLARI ---
# Not: Logic senin kodunla birebir aynı tutuldu.

def classify_input(state: State):
    router_prompt = """
    Sen bir yönlendirme asistanısın. Kullanıcının sorusunu analiz et.
    
    Veritabanı Tabloları:
    {table_info}
    
    Karar Mantığı:
    1. Eğer soru, veritabanındaki tablolardan veri çekmeyi gerektiriyorsa (örn: "kaç inek var", "dünkü süt üretimi") -> "SQL" cevabını ver.
    2. Eğer soru genel bilgi, sohbet veya veritabanında olmayan bir konuysa -> "GENERAL" cevabını ver.
    
    Sadece "SQL" veya "GENERAL" kelimesini döndür.
    
    Soru: {question}
    """
    prompt = ChatPromptTemplate.from_template(router_prompt)
    chain = prompt | llm | StrOutputParser()
    
    if db is None:
         return {"classification": "general"}

    response = chain.invoke({
        "table_info": db.get_table_info(),
        "question": state["question"]
    })
    
    decision = response.strip().upper()
    return {"classification": "sql" if "SQL" in decision else "general"}

def write_query(state: State):
    system_prompt = """
    Sen PostgreSQL konusunda uzmanlaşmış, kıdemli bir Veritabanı Mühendisisin.
    Görevin: Çiftçinin doğal dilde sorduğu soruları, verilen şemaya uygun, en optimize ve hatasız SQL sorgularına çevirmektir.

    --- VERİTABANI ŞEMASI VE KURALLAR ---
    {table_info}

    --- KRİTİK POSTGRESQL SÖZDİZİMİ KURALLARI ---
    1. **ÇIKTI FORMATI:** Sadece ve sadece saf SQL kodu üret. Markdown, tırnak işareti EKLEME.
    2. **ZAMAN:** "Bugün"=CURRENT_DATE, "Dün"=CURRENT_DATE - INTERVAL '1 day'
    3. **AGGREGATION:** FILTER (WHERE ...) içinde aggregate kullanma.
    4. **METİN:** İnek isimleri için `ILIKE` kullan.
    5. **UNION:** UNION kullanırken SELECT bloklarını paranteze al: (SELECT ...) UNION ALL (SELECT ...).
    """
    
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        ("user", "Soru: {input}")
    ])
    
    chain = query_prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "table_info": db.get_table_info(), 
        "input": state["question"]
    })
    
    clean_query = result.strip().replace("```sql", "").replace("```", "").strip()
    return {"query": clean_query}

def execute_query(state: State):
    if not state.get("query"):
        return {"result": "Sorgu oluşturulamadı."}
        
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    try:
        res = execute_query_tool.invoke(state["query"])
    except Exception as e:
        res = f"Hata oluştu: {str(e)}"
    return {"result": res}

def generate_sql_answer(state: State):
    prompt_template = """
    Sen **Süt Sihirbazı**'sın. Veritabanından gelen şu sonucu çiftçiye açıkla.
    Soru: {question}
    Sonuç: {result}
    Samimi ve net ol.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": state["question"], "result": state["result"]})
    return {"answer": response}

def generate_general_answer(state: State):
    prompt_template = """
    Sen **Süt Sihirbazı**'sın. Çiftçilere yardım eden neşeli bir yapay zeka asistanısın.
    Soru: {question}
    Samimi, yardımsever bir dille cevap ver.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": state["question"]})
    return {"answer": response}

def route_decision(state: State):
    return "write_query" if state["classification"] == "sql" else "generate_general_answer"

# --- UYGULAMA YAŞAM DÖNGÜSÜ (LIFESPAN) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Uygulama başlarken çalışacak kodlar
    global db, llm, app_workflow
    print("Süt Sihirbazı Başlatılıyor...")
    
    db = get_database()
    llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)
    
    # Graph Kurulumu
    workflow = StateGraph(State)
    workflow.add_node("classify", classify_input)
    workflow.add_node("write_query", write_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("generate_sql_answer", generate_sql_answer)
    workflow.add_node("generate_general_answer", generate_general_answer)

    workflow.add_edge(START, "classify")
    workflow.add_conditional_edges(
        "classify",
        route_decision,
        {"write_query": "write_query", "generate_general_answer": "generate_general_answer"}
    )
    workflow.add_edge("write_query", "execute_query")
    workflow.add_edge("execute_query", "generate_sql_answer")
    workflow.add_edge("generate_sql_answer", END)
    workflow.add_edge("generate_general_answer", END)
    
    app_workflow = workflow.compile()
    print("Graph derlendi ve hazır.")
    
    yield
    # Uygulama kapanırken çalışacak kodlar (varsa db kapatma vs.)
    print("Süt Sihirbazı Kapatılıyor...")

# --- FASTAPI APP ---
app = FastAPI(title="Süt Sihirbazı API", lifespan=lifespan)

@app.post("/query", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Çiftçinin sorusunu alır, işler ve cevabı döndürür.
    """
    if not app_workflow:
        raise HTTPException(status_code=500, detail="Workflow başlatılamadı.")

    # invoke yerine ainvoke (asenkron) kullanarak bloklamayı önlüyoruz
    inputs = {"question": request.question}
    try:
        # LangGraph invoke sonucunu al
        result = await app_workflow.ainvoke(inputs)
        
        return QueryResponse(
            question=request.question,
            classification=result.get("classification", "unknown"),
            answer=result.get("answer", "Cevap üretilemedi."),
            sql_query=result.get("query"),  # Sadece SQL modundaysa dolu gelir
            sql_result=str(result.get("result")) if result.get("result") else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
def health_check():
    return {"status": "active", "db_connected": db is not None}

# Doğrudan çalıştırma için (python main.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)