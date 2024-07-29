from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import base64

def convert_bytes_to_base64(image_bytes):
    encoded_string = base64.b64encode(image_bytes).decode("utf-8")
    return "data:image/jpeg;base64," + encoded_string

def handle_images(image_bytes_list, user_message):
    chat_handler = Llava15ChatHandler(clip_model_path = "./models/llava/mmproj-model-f16.gguf") ##mention the model path here make sure its in gguf format 
    llm = Llama(
        model_path = "./models//llava/ggml-model-q5_k.gguf", #mention the model path here make sure its in gguf format 
        chat_handler = chat_handler,
        logits_all = True,
        n_ctx = 1024
    )
    
    messages = [
        {"role": "system", "content": "You are a assistant who can perfectly describe the image or diagrams given to you in a way that even 8 year old can understand it, and your name is Yaar"}
    ]
    
    for image_bytes in image_bytes_list:
        image_base64 = convert_bytes_to_base64(image_bytes)
        messages.append({
            "role" : "user",
            "content" : [
                {"type": "image_url", "image_url": {"url": image_base64}},
                {"type": "text", "text": user_message}
            ]
        })
    
    output = llm.create_chat_completion(messages=messages)
    
    # Concatenate the outputs for each image
    combined_output = ""
    for choice in output["choices"]:
        combined_output += choice["message"]["content"]
    
    return [combined_output]
