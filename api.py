from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional
from rag import rag_app
import uvicorn
import whisper
import os
import tempfile

app = FastAPI()



class QueryRequest(BaseModel):
    question: str
    
class QueryResponse(BaseModel):
    answer: str
    classification: str
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None

class VoiceQueryResponse(BaseModel):
    answer: str
    classification: str
    transcription: str  
    sql_query: Optional[str] = None
    sql_result: Optional[str] = None

class TranscriptionResponse(BaseModel):
    text: str
    success: bool
    
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

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Ses dosyasını alır ve Whisper ile Türkçe transkripsiyonu yapar
    """
    print("Whisper modeli yükleniyor...")
    whisper_model = whisper.load_model("medium")
    print("Whisper modeli hazır!")

    try:
        # Desteklenen ses formatlarını kontrol et
        allowed_extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']
        file_ext = os.path.splitext(audio.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Desteklenmeyen dosya formatı. İzin verilenler: {', '.join(allowed_extensions)}"
            )
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            # Yüklenen dosyayı geçici dosyaya yaz
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Whisper ile transkribe et
            result = whisper_model.transcribe(
                temp_file_path, 
                language="tr",  # Türkçe
                fp16=False
            )
            
            transcribed_text = result["text"].strip()
            
            return TranscriptionResponse(
                text=transcribed_text,
                success=True
            )
            
        finally:
            # Geçici dosyayı temizle
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transkripsiyon hatası: {str(e)}")

@app.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(audio: UploadFile = File(...)):
    """
    Ses dosyasını transkribe edip direkt olarak query endpoint'ine gönderir
    """
    try:
        # Önce transkripsiyonu yap
        transcription_result = await transcribe_audio(audio)
        
        if not transcription_result.success or not transcription_result.text:
            raise HTTPException(status_code=400, detail="Ses metne dönüştürülemedi")
        
        # Transkripsiyonu query endpoint'ine gönder
        state = {"question": transcription_result.text}
        final_state = rag_app.invoke(state)
        
        response = VoiceQueryResponse(
            answer=final_state["answer"],
            classification=final_state["classification"],
            transcription=transcription_result.text,  # Yeni eklendi
            sql_query=final_state.get("query"),
            sql_result=final_state.get("result")
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sorgu işleme hatası: {str(e)}")

