from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings


import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


db = FAISS.load_local("./faiss_index", embeddings, allow_dangerous_deserialization=True)


retriever = db.as_retriever(k=4)
docs = retriever.invoke("What are ngrams?")

query = "What is N-Gram"
k = 3
similar_documents = db.similarity_search(query, k=k)

# Display the results
for i, doc in enumerate(similar_documents, start=1):
    print(f"Document {i}:")
    print(f"Content: {doc.page_content}\n")


base_model = "mistralai/Mistral-7B-Instruct-v0.2"


# Load the Mistral 7B model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map={"": 0}
)
model.config.use_cache = True
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token


adapter_location = "./model"
peft_model = PeftModel.from_pretrained(model, adapter_location)

query = "How does the choice of activation function impact the efficiency of backpropagation in neural networks?"
k = 3
similar_documents = db.similarity_search(query, k=k)

documents = []
for i, doc in enumerate(similar_documents, start=1):
    documents.append(doc.page_content)


def stream(user_prompt):
    runtimeFlag = "cuda:0"
    system_prompt = (
        "You are a helpful teaching assistant for natural language processing.\n"
    )

    B_INST, E_INST = "[INST]", "[/INST]"

    prompt = f"{system_prompt}{B_INST}{user_prompt.strip()}\n{E_INST}"

    inputs = tokenizer([prompt], return_tensors="pt").to(runtimeFlag)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    _ = model.generate(**inputs, streamer=streamer, max_new_tokens=1024)


documents = " ".join(documents).replace("\n", " ")
prompt = f"""

Context:
{documents}

Using the Context provide a detailed explaination to the question. Do not reference individual chapters but answer succinctly to the point.
{query}
"""


print("Prompt:", prompt)


stream(prompt)
