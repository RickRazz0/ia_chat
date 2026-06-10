import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai import types
from dotenv import load_dotenv

app = FastAPI(
    title="Backend Chatbot Gemini",
    description="API robusta para conectar um frontend ao Gemini 2.5 Flash com MCP",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# O SDK do Google procura por GOOGLE_API_KEY, então mapeamos GEMINI_API_KEY se existir
api_key = os.environ.get("GEMINI_API_KEY")
model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-flash-latest')
    except Exception as e:
        print(f"AVISO CRÍTICO: Falha ao configurar o cliente Gemini com a API Key. Detalhe: {e}")
        model = None
else:
    print("AVISO: GEMINI_API_KEY não encontrada no ambiente.")



# --- Modelos de Dados ---
class ChatRequest(BaseModel):
    mensagem: str


class ChatResponse(BaseModel):
    resposta: str

# --- Funções Auxiliares (MCP) ---
async def get_mcp_context(query: str) -> str:
    """Espaço reservado para a conexão com o Servidor MCP."""
    return ""

# --- Rotas da API ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not model:
        raise HTTPException(
            status_code=500, 
            detail="Servidor não possui chave de API configurada ou falhou ao inicializar."
        )

    try:
        mcp_context = await get_mcp_context(request.mensagem)

        if mcp_context:
            final_prompt = f"Contexto adicional:\n{mcp_context}\n\nMensagem do usuário: {request.mensagem}"
        else:
            final_prompt = request.mensagem
        
        response = await model.generate_content_async(
            final_prompt,
            generation_config=types.GenerationConfig(
                temperature=0.7,
            )
        )

        return ChatResponse(resposta=response.text)

    except Exception as e:
        print(f"Erro na comunicação com o Gemini: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro na API da Google: {str(e)}")


@app.get("/health")
async def health_check():
    status_gemini = "OK" if model else "Sem API Key"
    return {"status": "Servidor rodando", "gemini_client": status_gemini}
