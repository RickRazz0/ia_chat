import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

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
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

try:
    client = genai.Client()
except Exception as e:
    print(f"AVISO CRÍTICO: Falha ao inicializar o cliente Gemini.")
    print(f"Verifique se o arquivo .env existe e contém GEMINI_API_KEY. Detalhe: {e}")
    client = None



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
    # Verifica se o cliente inicializou corretamente lá em cima
    if not client:
        raise HTTPException(
            status_code=500, 
            detail="Servidor não possui chave de API configurada. Verifique o arquivo .env."
        )

    try:
        mcp_context = await get_mcp_context(request.mensagem)

        if mcp_context:
            final_prompt = f"Contexto adicional:\n{mcp_context}\n\nMensagem do usuário: {request.mensagem}"
        else:
            final_prompt = request.mensagem

        # --- 1. Correção do Async (Usando client.aio) ---
        # Agora o servidor não bloqueia esperando a resposta da Google!
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=final_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
            )
        )

        return ChatResponse(resposta=response.text)

    except Exception as e:
        # Imprime o erro real no terminal do VS Code para você investigar
        print(f"Erro na comunicação com o Gemini: {str(e)}")
        # Devolve o erro real para o Swagger/Frontend ao invés de um 500 genérico
        raise HTTPException(status_code=500, detail=f"Erro na API da Google: {str(e)}")

@app.get("/health")
async def health_check():
    status_gemini = "OK" if client else "Sem API Key"
    return {"status": "Servidor rodando", "gemini_client": status_gemini}
