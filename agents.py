import dspy
from common import (
    LM_MODEL_TEACHER,
    LM_MODEL_STUDENT,
    diabetes_vector_search_tool,
    copd_vector_search_tool,
)


class ReActSignature(dspy.Signature):
    """You are a helpful assistant. Answer user's question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class DiabetesAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        # init LLM
        self.lm = dspy.LM(
            LM_MODEL_STUDENT,
            temperature=0.3,
            max_tokens=-1,
            cache=False,
        )
        dspy.configure(lm=self.lm)
        self.agent = dspy.ReAct(ReActSignature, tools=[diabetes_vector_search_tool])

    def forward(self, question: str):
        return self.agent(question=question)


# Instantiate the COPD expert agent
class COPDAgent(dspy.Module):
    def __init__(self):
        self.lm = dspy.LM(
            LM_MODEL_STUDENT,
            temperature=0.3,
            max_tokens=-1,
            cache=False,
        )
        dspy.configure(lm=self.lm)
        self.copd_agent = dspy.ReAct(ReActSignature, tools=[copd_vector_search_tool])

    def forward(self, question: str):
        return self.copd_agent(question=question)


# Load or Make optimized agents
def 

# Wrap the domain agents as callable tools for the lead agent
# Prefer the optimized Diabetes agent if available; otherwise fallback to baseline `react`.


def ask_diabetes(question: str) -> str:
    """Call the Diabetes expert agent and return its answer text."""
    assert optimized_diabetes_agent
    pred = optimized_diabetes_agent(question=question)
    return pred.answer


def ask_copd(question: str) -> str:
    """Call the COPD expert agent and return its answer text."""
    assert optimized_copd_agent
    pred = optimized_copd_agent(question=question)
    return pred.answer


# Lead ReAct agent that can call sub-agents as tools
class LeadReAct(dspy.Module):
    def __init__(self):
        self.lm = dspy.LM(
            LM_MODEL_STUDENT,
            temperature=0.3,
            max_tokens=-1,
            cache=False,
        )
        dspy.configure(lm=self.lm)
        self.lead_react = dspy.ReAct(ReActSignature, tools=[ask_diabetes, ask_copd])

    def forward(self, question: str):
        return self.lead_react(question=question)


lead_react = LeadReAct()
