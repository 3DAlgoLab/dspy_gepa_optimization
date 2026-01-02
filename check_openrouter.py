# %%
import dspy
import os

from dotenv import load_dotenv

LM_MODEL_STUDENT = "openrouter/nvidia/nemotron-nano-12b-v2-vl:free"
LM_MODEL_TEACHER = "openrouter/openai/gpt-oss-120b:free"

load_dotenv()

# %%
question = "What is a language model in one sentence?"
lm = dspy.LM(
    LM_MODEL_TEACHER,  # LiteLLM route for OpenRouter models    
    model_type="chat",
    temperature=0.5,
    max_tokens=-1,
)

print(lm(question))

#%%
lm_student = dspy.LM(
    LM_MODEL_STUDENT,    
    model_type="chat",
    temperature=0.5,
    max_tokens=-1,
)

print("student:")
print(lm_student(question))