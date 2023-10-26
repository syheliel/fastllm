import chainlit as cl
from fastllm_pytools import llm
model = llm.model("Qwen-7B-Chat-fp16.flm")

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session

    # Store the chain in the user session
    # cl.user_session.set("llm_chain", llm_chain)
    # print(2)
    pass

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    
    msg = cl.Message(
        content="",
    )

    # Call the chain asynchronously
    # my_prompt = "This is my prompt " + message.content
    for response in model.stream_response(message.content):
        await msg.stream_token(response)
        
    await msg.send()
