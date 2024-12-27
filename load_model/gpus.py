from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
from typing import List, Optional
import time
import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from models.project import TNTProject
import os
import torch

# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

#meta-llama/Llama-3.3-70B-Instruct
dev = False

if dev == False:
    model_name = 'cognitivecomputations/dolphin-2.9-llama3-8b'

    # Load model and tokenizer with optimized settings for 8B model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # Changed to bfloat16 for better numerical stability
        load_in_8bit=False,  # Disabled 8-bit quantization since we're using bfloat16
        offload_folder="offload",
        max_memory={i: "12GiB" for i in range(torch.cuda.device_count())},  # Dynamically set per available GPU
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
else :
    model_name = ''
    model = None
    tokenizer = None
    pipe = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    conversation_id: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    max_length: int = 16000
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    do_sample: bool = True
    truncation: bool = True

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

def get_gpus(request: ChatRequest, project: TNTProject):
    conversation_id = project.create_conversation(request.conversation_id)

    try:
        # Format messages for the model
        new_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        formatted_messages = project.get_conversation_messages(conversation_id)
        formatted_messages += new_messages

        for msg in new_messages:
            project.add_message_to_conversation(conversation_id, msg['role'], msg['content'])
        
        # Generate response
        response = None

        if dev == False:
            formatted_prompt = format_prompt(formatted_messages)
            response = pipe(
                formatted_prompt,
                max_length=request.max_length,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
                truncation=request.truncation,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Extract the generated text from the response
            generated_text = response[0]['generated_text']
            
            # Extract only the last assistant's message
            # Find the last assistant section
            parts = generated_text.split("<|im_start|>assistant\n")
            if len(parts) > 1:
                last_response = parts[-1].split("<|im_end|>")[0].strip()
            else:
                last_response = ""
                
            generated_text = last_response

        # Create response object
        chat_response = ChatResponse(
            id=conversation_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(formatted_messages),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(formatted_messages) + len(generated_text.split())
            }
        )

        project.add_message_to_conversation(conversation_id, "assistant", generated_text)

        # Serialize the response to JSON manually using json.dumps
        return JSONResponse(content=json.dumps(chat_response.model_dump()))  # Serialize and return as JSON response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_prompt(formatted_messages):
    prompt = ""
    
    # Add system message if present
    system_msg = next((msg for msg in formatted_messages if msg['role'] == 'system'), None)
    if system_msg:
        prompt += f"<|im_start|>system\n{system_msg['content']}<|im_end|>\n"
    
    # Add all user/assistant messages in order
    for msg in formatted_messages:
        if msg['role'] in ['user', 'assistant']:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    
    # Add final assistant token to generate response
    prompt += "<|im_start|>assistant\n"
    
    return prompt