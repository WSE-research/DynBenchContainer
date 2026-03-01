#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

DEBUG = False

uvicorn_error = logging.getLogger("uvicorn.error")
logger = logging.getLogger()
logger.parent = uvicorn_error
logger.setLevel(logging.DEBUG) if DEBUG else logger.setLevel(level=logging.INFO)

import argparse

import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from startup import LANGUAGES

from core import create_question_query
from core import feedback_collection

from startup import call_LLM, LLM_URL, KEY


app = FastAPI()


def main():
    parser = argparse.ArgumentParser(description="Transform question and query over Wikidata by replacing one entity.")

    # Add arguments
    parser.add_argument('-q', '--question', type=str, required=True, help='The question to process')
    parser.add_argument('-r', '--query', type=str, required=True, help='The query to process')
    parser.add_argument('-m', '--model', type=str, required=True, help='The LLM model to use')
    parser.add_argument(
        '-l', '--lang', 
        type=str, 
        choices=[v for v in LANGUAGES.values() if len(v) < 4],
        help='Language of the question')
    parser.add_argument(
        '-c', '--complexity',
        type=str,
        choices=['easy', 'normal', 'hard', 'random'],
        default='normal',
        help='The complexity level: easy, normal, or hard (default: normal)'
    )

    args = parser.parse_args()

    question = args.question
    query = args.query
    complexity = args.complexity
    lang = args.lang
    model = args.model

    new_question, new_query, old_pagerank, new_pagerank = create_question_query(query, question, model, lang, complexity, None)

    logger.info(f'Original Question: {question}')
    logger.info(f'Original Query: {query}')
    logger.info(f'Transformed Question: {new_question}')
    logger.info(f'Transformed Query: {new_query}')
    logger.info(f'Old PageRank: {old_pagerank}')
    logger.info(f'New PageRank: {new_pagerank}')


app = FastAPI()


class TransformRequest(BaseModel):
    question: str
    query: str
    model: str
    lang: str
    complexity: str = "normal"
    checks: list[str] | None


class TransformResponse(BaseModel):
    original_question: str
    original_query: str
    transformed_question: str | None
    transformed_query: str | None
    old_pagerank: float | None
    new_pagerank: float | None
    extra: dict | None = None


class FeedbackRequest(BaseModel):
    inputs: list[str]
    outputs: list[str]
    rating: int


class DetectLanguageRequest(BaseModel):
    text: str
    model: str


class DetectLanguageResponse(BaseModel):
    detected_language: str | None
    confidence: float | None


# lock to prevent multply calls
transform_lock = threading.Lock()

@app.post("/detect_language")
def detect_language(request: DetectLanguageRequest) -> DetectLanguageResponse:
    """Detect the language of a given text using an LLM.
    Args:
        request (DetectLanguageRequest): The request body containing text and model.
    """
    prompt = f"""Detect the language of the following text.
Return only the language code (e.g., en, ru, de, fr, es) in a JSON object with key "language_code".

Text: {request.text}"""

    result = call_LLM(
        url=LLM_URL,
        key=KEY,
        model=request.model,
        prompt=prompt,
        temp=0.0,
        max_tokens=10
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to detect language")
    
    try:
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not response_text:
            response_text = result.get("text", "")
        
        # Extract language code from response
        import json
        try:
            # Try parsing as JSON first
            parsed = json.loads(response_text)
            lang_code = parsed.get("language_code", "")
        except json.JSONDecodeError:
            # If not JSON, try to extract language code directly
            lang_code = response_text.strip()
        
        # Validate it's a 2-letter language code
        if len(lang_code) == 2 and lang_code.isalpha():
            return DetectLanguageResponse(detected_language=lang_code.lower(), confidence=0.95)
        else:
            # Fallback: try to match with known languages
            for code, name in LANGUAGES.items():
                if lang_code.lower() in name.lower() or name.lower() in lang_code.lower():
                    return DetectLanguageResponse(detected_language=code, confidence=0.8)
            
            raise HTTPException(status_code=400, detail=f"Could not parse valid language code from response: {response_text}")
    
    except Exception as e:
        logger.error(f"Error parsing language detection response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transform", response_model=TransformResponse)
def transform_endpoint(request: TransformRequest) -> TransformResponse:
    """Transform the question and query by replacing one entity.
    Args:
        request (TransformRequest): The request body containing question, query, model, lang, and complexity.
    """
    if request.lang not in LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {request.lang} (supported: {', '.join(LANGUAGES.keys())})"
        )
    
    if request.complexity not in ['easy', 'normal', 'hard', 'random']:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid complexity: {request.complexity} (supported: easy, normal, hard, random)"
        )
    
    if not transform_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=503,
            detail="System is busy. Please try again later."
        )
            
    try:
        new_question, new_query, old_pagerank, new_pagerank = create_question_query(
            request.query,
            request.question,
            request.model,
            request.lang,
            request.complexity,
            request.checks
        )
        
        return TransformResponse(
            original_question=request.question,
            original_query=request.query,
            transformed_question=new_question,
            transformed_query=new_query,
            old_pagerank=old_pagerank,
            new_pagerank=new_pagerank
        )
    
    except Exception as e:
        logger.error(f"Error in transform endpoint: {e} \n(request: {request})")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Always release the lock so the endpoint becomes available again
        transform_lock.release()


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    try:
        feedback_doc = {
            "inputs": feedback.inputs,
            "outputs": feedback.outputs,
            "rating": feedback.rating,
        }

        result = await feedback_collection.insert_one(feedback_doc)

        return {
            "message": "Feedback stored successfully",
            "feedback_id": str(result.inserted_id)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    """Redirect to the documentation."""
    return RedirectResponse(url='/docs')


@app.get("/health")
def health_check():
    """Check the health of the API."""
    return {"status": "healthy"}


if DEBUG:
    query = 'SELECT ?answer WHERE { wd:Q14452 wdt:P17 ?answer }'
    create_question_query(
        query,
        'Which country does the famous Easter island belong to?',
        'Gemma3:27b',
        'en', 
        'normal',
        None
    )
else:
    if __name__ == "__main__":
        main()

