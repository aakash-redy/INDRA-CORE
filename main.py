# ==========================================
# HEXAWATTS SORA - PERFECT BACKEND (main.py)
# ==========================================

# 1. ALL IMPORTS (Zero missing dependencies)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# 2. ENVIRONMENT & API SETUP
load_dotenv()

# Verify API keys are present (Helps debug if .env is missing)
if not os.getenv("GEMINI_API_KEY") or not os.getenv("SUPABASE_URL"):
    print("⚠️ WARNING: Missing API keys in .env file!")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# 3. FASTAPI APP INIT
app = FastAPI(title="Hexawatts Sora API")

# 4. CORS MIDDLEWARE (Allows your React frontend on port 5173 to talk to this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. DATA VALIDATION MODEL
class QueryRequest(BaseModel):
    question: str
    domain: str
    year: str # Expects "2026", "2027", or "Both" from your React UI

# 6. THE CORE RAG ENDPOINT
@app.post("/ask_sora")
async def ask_sora(req: QueryRequest):
    try:
        print(f"Received query: '{req.question}' | Domain: {req.domain} | Year: {req.year}")
        
        # Step A: Convert the user's question into a mathematical vector
        query_vector = genai.embed_content(
            model="models/text-embedding-004",
            content=req.question,
            task_type="retrieval_query",
        )['embedding']

        # Step B: Search the Supabase Vector Database
        response = supabase.rpc(
            'match_rules',
            {
                'query_embedding': query_vector, 
                'match_threshold': 0.5, 
                'match_count': 10,  # Grabbing 10 rules to ensure we don't miss context
                'rule_year': req.year 
            }
        ).execute()

        matches = response.data
        
        # Step C: Handle cases where the rule isn't found
        if not matches:
            return {
                "answer": f"I couldn't find a specific rule regarding this in the {req.year} rulebook.", 
                "sources": []
            }

        # Step D: Format the chunks for the AI to read
        context = "\n\n".join([f"[Year: {m['year']}] Rule {m['rule_id']}: {m['content']}" for m in matches])
        
        # Step E: Handle the "Compare Both" logic
        comparison_instruction = ""
        if req.year == "Both":
            comparison_instruction = "The user is asking to compare rules between 2026 and 2027. Explicitly highlight any differences in dimensions, materials, or requirements between the two years. If there are no changes, state that the rule remains identical."

        # Step F: The Master Prompt
        prompt = f"""
        You are Sora, the strict and precise AI Scrutineer for Hexawatts Racing. 
        The user is asking a question related to the '{req.domain}' domain.
        
        YOUR DIRECTIVES:
        1. Answer the question using ONLY the provided Rulebook Context below.
        2. If the context does not contain the exact answer, explicitly state: "I cannot find the exact rule for this in the provided context." Do NOT guess.
        3. Maintain a professional, engineering-focused tone.
        {comparison_instruction}
        
        User Question: {req.question}
        
        Rulebook Context:
        {context}
        """
        
        # Step G: Generate Answer (Temperature 0.1 stops hallucinations)
        # FIX: 'gemini-3.1-pro' does not exist in the Google Generative AI model
        # list — it was causing every /ask_sora request to fail with 500.
        # 'gemini-2.0-flash' is fast, capable, and matches the model family used
        # in apps/api/index.ts (PRIMARY roster), so behavior is consistent
        # across both backends. The model is overridable via SORA_MODEL_NAME env.
        MODEL_NAME = os.getenv("SORA_MODEL_NAME", "gemini-2.0-flash")
        try:
            model = genai.GenerativeModel(MODEL_NAME)
        except Exception as model_err:
            # If the preferred model isn't available on this account, fall back
            # to a long-stable model (gemini-1.5-pro) instead of crashing.
            print(f"⚠️  Model '{MODEL_NAME}' unavailable ({model_err}); falling back to gemini-1.5-pro.")
            model = genai.GenerativeModel("gemini-1.5-pro")
        answer = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        
        # Step H: Extract and sort unique sources (e.g., "T1.1.5 (2026)")
        sources = sorted(list(set([f"{m['rule_id']} ({m['year']})" for m in matches])))
        
        return {
            "answer": answer.text,
            "sources": sources
        }

    except Exception as e:
        # This prints the exact error in your terminal so you aren't guessing
        print(f"\n🔥 SORA ERROR DETAILS: {str(e)}\n") 
        raise HTTPException(status_code=500, detail=str(e))