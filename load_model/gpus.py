from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pydantic import BaseModel
from typing import List, Optional
import time
import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from models.project import TNTProject

#meta-llama/Llama-3.3-70B-Instruct
dev = False

if dev == False:
    model_name = 'cognitivecomputations/dolphin-2.9-llama3-8b'

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        offload_folder="offload"
    )
    # meta-llama/Llama-3.3-70B-Instruct
    # cognitivecomputations/dolphin-2.9.2-qwen2-72b
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
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
            response = pipe(formatted_messages, 
                       max_length=4096,
                       max_new_tokens=request.max_tokens,
                       temperature=request.temperature,
                       truncation=True)
        # Extract the last assistant message from the generated response
        generated_text = ""
        if isinstance(response, list) and len(response) > 0:
            for message in response[0]["generated_text"]:
                if message["role"] == "assistant":
                    generated_text = message["content"]  # Store the assistant's message

            # If no assistant message found, set generated_text to an empty string
            if not generated_text:
                generated_text = ""

        else:
            generated_text = ""

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