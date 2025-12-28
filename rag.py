from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, END, StateGraph
import os 
from dotenv import load_dotenv



load_dotenv()


def get_database():
    try:
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:5432/{db_name}"
        db=SQLDatabase.from_uri(db_uri)
        return db
    except Exception as e:
       print(f"Veritabanı bağlantı hatası: {e}")
       return None

db = get_database()


llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0,base_url="http://localhost:11434")


class State(TypedDict):
    question: str
    classification: str 
    query: str
    result: str
    answer: str

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

    --- KRİTİK POSTGRESQL SÖZDİZİMİ KURALLARI (BUNLARA KESİNLİKLE UY) ---
    1. **ÇIKTI FORMATI:** Sadece ve sadece saf SQL kodu üret. Markdown, tırnak işareti veya açıklama metni EKLEME.
    2. **ZAMAN KAVRAMI:** - "Bugün" = CURRENT_DATE
       - "Dün" = CURRENT_DATE - INTERVAL '1 day'
    3. **AGGREGATION:** `FILTER (WHERE ...)` bloğu içinde aggregate fonksiyon (SUM, AVG) kullanma.
    4. **METİN ARAMALARI:** İnek isimleri için her zaman `ILIKE` kullan.
    5. **SÜT HESABI:** `sut` tablosunda `gunluk_sagim` zaten hesaplanmış bir kolondur. Toplama işlemi yapma.
    
    6. **UNION VE LIMIT KULLANIMI (ÇOK ÖNEMLİ):**
       - Eğer "En yüksek X ve En düşük Y" gibi bir soru gelirse ve `UNION` kullanman gerekirse;
       - Her iki `SELECT` sorgusunu da **MUTLAKA PARANTEZ İÇİNE AL**.
       - PostgreSQL, parantezsiz `UNION` içindeki `ORDER BY` ve `LIMIT` ifadelerinde hata verir.
       - DOĞRU: `(SELECT ... ORDER BY ... LIMIT 5) UNION ALL (SELECT ... ORDER BY ... LIMIT 5)`
       - YANLIŞ: `SELECT ... ORDER BY ... LIMIT 5 UNION SELECT ...`

    --- ÖRNEK SENARYOLAR (FEW-SHOT LEARNING) ---
    
    Kullanıcı: "Sarıkız'ın dünkü süt verimi nedir?"
    SQL: 
    SELECT s.gunluk_sagim 
    FROM sut s 
    JOIN inekler i ON s.inek_id = i.inek_id 
    WHERE i.inek_name ILIKE '%Sarıkız%' AND s.sagim_tarihi = CURRENT_DATE - INTERVAL '1 day';

    Kullanıcı: "Süt verimi en yüksek ve en düşük 3 ineği getir"
    SQL:
    (SELECT i.inek_name, SUM(s.gunluk_sagim) as toplam_sut
     FROM sut s JOIN inekler i ON s.inek_id = i.inek_id
     GROUP BY i.inek_id, i.inek_name
     ORDER BY toplam_sut DESC
     LIMIT 3)
    UNION ALL
    (SELECT i.inek_name, SUM(s.gunluk_sagim) as toplam_sut
     FROM sut s JOIN inekler i ON s.inek_id = i.inek_id
     GROUP BY i.inek_id, i.inek_name
     ORDER BY toplam_sut ASC
     LIMIT 3);

    Kullanıcı: "Geçen aya göre süt verimi düşen inekler hangileri?"
    SQL:
    WITH gecen_ay AS (
        SELECT inek_id, AVG(gunluk_sagim) as ort_verim
        FROM sut
        WHERE sagim_tarihi >= date_trunc('month', CURRENT_DATE - INTERVAL '1 month') 
          AND sagim_tarihi < date_trunc('month', CURRENT_DATE)
        GROUP BY inek_id
    ),
    bu_ay AS (
        SELECT inek_id, AVG(gunluk_sagim) as ort_verim
        FROM sut
        WHERE sagim_tarihi >= date_trunc('month', CURRENT_DATE)
        GROUP BY inek_id
    )
    SELECT i.inek_name, b.ort_verim as bu_ay, g.ort_verim as gecen_ay
    FROM bu_ay b
    JOIN gecen_ay g ON b.inek_id = g.inek_id
    JOIN inekler i ON b.inek_id = i.inek_id
    WHERE b.ort_verim < g.ort_verim;
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
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    try:
        res = execute_query_tool.invoke(state["query"])
    except Exception as e:
        res = f"Hata oluştu: {str(e)}"
    return {"result": res}

def generate_sql_answer(state: State):
    prompt_template = """
    Sen **Süt Sihirbazı**'sın. Veritabanından gelen şu sonucu doğal dilde anlaşılır bir tonda  çiftçiye açıkla.
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


workflow = StateGraph(State)
workflow.add_node("classify", classify_input)
workflow.add_node("write_query", write_query)
workflow.add_node("execute_query", execute_query)
workflow.add_node("generate_sql_answer", generate_sql_answer)
workflow.add_node("generate_general_answer", generate_general_answer)

workflow.add_edge(START, "classify")

def route_decision(state: State):
    return "write_query" if state["classification"] == "sql" else "generate_general_answer"

workflow.add_conditional_edges(
    "classify",
    route_decision,
    {"write_query": "write_query", "generate_general_answer": "generate_general_answer"}
)

workflow.add_edge("write_query", "execute_query")
workflow.add_edge("execute_query", "generate_sql_answer")
workflow.add_edge("generate_sql_answer", END)
workflow.add_edge("generate_general_answer", END)

rag_app = workflow.compile()