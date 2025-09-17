import time
import sys
from typing import Tuple
from langchain_google_genai import  ChatGoogleGenerativeAI
from app.prompts.prompt_library import PROMPT_REGISTRY
from app.telemetry.custom_exception import customException
from app.telemetry.custom_logger import CustomLogger

log=CustomLogger().get_logger(__name__)

class TextClassifier:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
    async def classify(self, text: str, prompt_type: str = "advanced") -> Tuple[str, float, str, int]:
        start_time = time.time()
        
        try:
                # Get prompt
            prompt_key = f"{prompt_type}_classification"
            prompt = PROMPT_REGISTRY[prompt_key]
            
            # Create chain and invoke
            chain = prompt | self.llm
            response = await chain.ainvoke({"text": text})
            
            # Parse response
            classification = self._parse_classification(response.content)
            latency_ms = int((time.time() - start_time) * 1000)
            
            
            log.info("classification has done")
            return classification, prompt_type, latency_ms
            
        except Exception as e:
            log.error(f"error during classification {e}")
            raise customException("error during classification",sys)
    
    def _parse_classification(self, response: str) -> str:
        # Extract classification from response
        response = response.lower().strip()
        if "toxic" in response:
            return "toxic"
        elif "spam" in response:
            return "spam"
        else:
            return "safe"