"""
LLM Chain utility for processing queries with batch support using LangChain
"""

import asyncio
import os
import time
from typing import List
import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# OPENROUTER_MODEL = "google/gemini-2.5-flash-lite"
OPENROUTER_MODEL = "google/gemini-2.0-flash-001"
# OPENROUTER_MODEL = "openai/gpt-4.1-nano"

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set. Please set it in your environment or .env file.")

# System prompt for insurance policy questions
SYSTEM_PROMPT = """
**System Prompt for Insurance Policy Document QA**

You are an expert AI assistant specializing in answering questions about insurance policy documents. Use only the provided context to respond to user inquiries. Provide concise, accurate, and comprehensive answers that address all relevant concepts from the context. If the answer is not in the context, state: "The information is not available in the provided document."

**Instructions:**
- Carefully review the provided context to identify all relevant details.
- Answer the user's question directly, ensuring the response is complete and covers all pertinent aspects from the context.
- If the information is unavailable in the context, clearly state so without speculating.
- Maintain a professional, clear, and formal tone.
- Do not generate charts, graphs, or images unless explicitly requested, and only in accordance with chart generation guidelines.
"""

def call_openrouter_api(prompt: str) -> str:
    """
    Synchronous function to call OpenRouter API with detailed timing
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        str: The response from the LLM
    """
    api_start = time.time()
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",  # Required for some models
            "X-Title": "Policy RAG System"
        }
        
        data = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 200,
            "temperature": 0.1,  # Low temperature for more consistent answers
            "top_p": 0.9
        }
        
        request_start = time.time()
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=30)
        request_time = time.time() - request_start
        
        logger.debug(f"🌐 OpenRouter API call took {request_time:.3f}s - Status: {response.status_code}")
        
        response.raise_for_status()
        
        parse_start = time.time()
        result = response.json()
        parse_time = time.time() - parse_start
        
        # Extract the generated text from the response
        try:
            extract_start = time.time()
            answer = result["choices"][0]["message"]["content"].strip()
            extract_time = time.time() - extract_start
            
            api_total = time.time() - api_start
            logger.debug(f"🔧 API breakdown - Request: {request_time:.3f}s, Parse: {parse_time:.3f}s, Extract: {extract_time:.3f}s, Total: {api_total:.3f}s")
            
            return answer
        except (KeyError, IndexError) as e:
            api_total = time.time() - api_start
            logger.error(f"❌ Error extracting response after {api_total:.3f}s: {e}")
            logger.error(f"Full response: {result}")
            return f"Error: Could not parse response - {str(result)}"
            
    except requests.exceptions.RequestException as e:
        api_total = time.time() - api_start
        logger.error(f"❌ Request error after {api_total:.3f}s: {e}")
        return f"Error: API request failed - {str(e)}"
    except Exception as e:
        api_total = time.time() - api_start
        logger.error(f"❌ Unexpected error after {api_total:.3f}s: {e}")
        return f"Error: {str(e)}"

async def process_chunk_with_llm_async(prompt: str) -> str:
    """
    Asynchronous wrapper for LLM processing with timing
    
    Args:
        prompt (str): The prompt to process
        
    Returns:
        str: The LLM response
    """
    async_start = time.time()
    loop = asyncio.get_event_loop()
    
    try:
        # Run the blocking API call in a thread pool executor
        response = await loop.run_in_executor(None, call_openrouter_api, prompt)
        async_total = time.time() - async_start
        logger.debug(f"🔄 Async LLM processing completed in {async_total:.3f}s")
        return response
    except Exception as e:
        async_total = time.time() - async_start
        logger.error(f"❌ Error in async LLM processing after {async_total:.3f}s: {e}")
        return f"Error: {str(e)}"

async def process_batches(tasks: List[asyncio.Task], batch_size: int = 3) -> List[str]:
    """
    Process tasks in batches to avoid rate limiting and improve performance
    
    Args:
        tasks (List[asyncio.Task]): List of async tasks to process
        batch_size (int): Number of tasks to process concurrently
        
    Returns:
        List[str]: List of responses
    """
    batch_start = time.time()
    results = []
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    
    logger.info(f"📊 Processing {len(tasks)} tasks in {total_batches} batches of size {batch_size}")
    
    for i in range(0, len(tasks), batch_size):
        batch_num = (i // batch_size) + 1
        batch = tasks[i:i + batch_size]
        
        batch_iteration_start = time.time()
        logger.info(f"🔄 Processing batch {batch_num}/{total_batches} with {len(batch)} tasks")
        
        try:
            # Process batch concurrently
            gather_start = time.time()
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            gather_time = time.time() - gather_start
            
            # Handle results and exceptions
            process_start = time.time()
            processed_results = []
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    error_msg = f"Error in task {i + j + 1}: {str(result)}"
                    logger.error(error_msg)
                    processed_results.append(error_msg)
                else:
                    processed_results.append(result)
            
            process_time = time.time() - process_start
            batch_iteration_time = time.time() - batch_iteration_start
            
            logger.info(f"✅ Batch {batch_num} completed in {batch_iteration_time:.3f}s (gather: {gather_time:.3f}s, process: {process_time:.3f}s)")
            
            results.extend(processed_results)
            
            # Add delay between batches to respect rate limits (except for the last batch)
            if i + batch_size < len(tasks):
                delay = 1.0  # 1 second delay between batches
                logger.debug(f"⏱️ Waiting {delay}s before next batch...")
                await asyncio.sleep(delay)
                
        except Exception as e:
            # If gather itself fails, add error messages for all tasks in the batch
            batch_iteration_time = time.time() - batch_iteration_start
            error_msg = f"Batch {batch_num} failed after {batch_iteration_time:.3f}s: {str(e)}"
            logger.error(error_msg)
            results.extend([error_msg] * len(batch))
    
    total_time = time.time() - batch_start
    logger.info(f"🏁 Completed processing all {len(tasks)} tasks in {total_time:.3f}s")
    return results

class LangChainLLMProcessor:
    """
    LangChain-based LLM processor for better integration and performance
    """
    
    def __init__(self):
        self.prompt_template = PromptTemplate(
            input_variables=["document", "question"],
            template="Based on the following document, answer the question:\n\nDocument:\n{document}\n\nQuestion:\n{question}"
        )
        
        # Create a runnable chain
        self.chain = (
            self.prompt_template 
            | RunnableLambda(self._format_and_call_llm)
            | StrOutputParser()
        )
    
    def _format_and_call_llm(self, formatted_prompt: str) -> str:
        """Format prompt and call LLM"""
        return call_openrouter_api(formatted_prompt.text)
    
    async def batch_process(self, documents: List[str], questions: List[str]) -> List[str]:
        """
        Process multiple questions using LangChain batch processing with detailed timing
        
        Args:
            documents (List[str]): List of document contexts
            questions (List[str]): List of questions to answer
            
        Returns:
            List[str]: List of answers
        """
        langchain_start = time.time()
        
        # Prepare inputs for batch processing
        prep_start = time.time()
        inputs = []
        for question in questions:
            # Use the same document context for all questions
            document_context = " ".join(documents) if isinstance(documents, list) else documents
            inputs.append({
                "document": document_context,
                "question": question
            })
        prep_time = time.time() - prep_start
        
        try:
            logger.info(f"🔗 Starting LangChain batch processing for {len(inputs)} questions")
            logger.debug(f"📋 Input preparation took {prep_time:.3f}s")
            
            # Use LangChain's batch method for efficient processing
            # Process in smaller batches to avoid overwhelming the API
            batch_size = 3
            all_results = []
            
            processing_start = time.time()
            
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(inputs) + batch_size - 1)//batch_size
                
                batch_start = time.time()
                logger.info(f"🔄 Processing LangChain batch {batch_num}/{total_batches}")
                
                # Create async tasks for this batch
                task_creation_start = time.time()
                tasks = [
                    asyncio.create_task(self._process_single_async(input_data))
                    for input_data in batch_inputs
                ]
                task_creation_time = time.time() - task_creation_start
                
                # Wait for batch completion
                gather_start = time.time()
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                gather_time = time.time() - gather_start
                
                # Handle results
                result_handling_start = time.time()
                for result in batch_results:
                    if isinstance(result, Exception):
                        all_results.append(f"Error: {str(result)}")
                    else:
                        all_results.append(result)
                result_handling_time = time.time() - result_handling_start
                
                batch_time = time.time() - batch_start
                logger.info(f"✅ LangChain batch {batch_num} completed in {batch_time:.3f}s")
                logger.debug(f"   📊 Task creation: {task_creation_time:.3f}s, Gather: {gather_time:.3f}s, Result handling: {result_handling_time:.3f}s")
                
                # Add delay between batches
                if i + batch_size < len(inputs):
                    await asyncio.sleep(1)
            
            processing_time = time.time() - processing_start
            total_time = time.time() - langchain_start
            
            logger.info(f"🏁 LangChain batch processing completed in {total_time:.3f}s")
            logger.info(f"   📊 Preparation: {prep_time:.3f}s, Processing: {processing_time:.3f}s")
            
            return all_results
            
        except Exception as e:
            total_time = time.time() - langchain_start
            logger.error(f"❌ Error in LangChain batch processing after {total_time:.3f}s: {e}")
            return [f"Error: {str(e)}"] * len(questions)
    
    async def _process_single_async(self, input_data: dict) -> str:
        """Process a single input asynchronously with timing"""
        single_start = time.time()
        loop = asyncio.get_event_loop()
        
        try:
            # Format the prompt
            format_start = time.time()
            formatted_prompt = self.prompt_template.format(**input_data)
            format_time = time.time() - format_start
            
            # Call LLM in thread pool
            llm_start = time.time()
            result = await loop.run_in_executor(None, call_openrouter_api, formatted_prompt)
            llm_time = time.time() - llm_start
            
            single_total = time.time() - single_start
            logger.debug(f"🔧 Single processing breakdown - Format: {format_time:.3f}s, LLM: {llm_time:.3f}s, Total: {single_total:.3f}s")
            
            return result
        except Exception as e:
            single_total = time.time() - single_start
            logger.error(f"❌ Error processing single question after {single_total:.3f}s: {str(e)}")
            return f"Error processing question: {str(e)}"

# Global instance
llm_processor = LangChainLLMProcessor()
