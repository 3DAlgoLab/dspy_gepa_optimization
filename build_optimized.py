import dspy
import os 
from dspy.evaluate import Evaluate

# Import GEPA (Generalized Evolutionary Prompt Adaptation) optimizer
from dspy.teleprompt import GEPA
# Import ScoreWithFeedback for detailed optimization feedback
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

from agents import DiabetesAgent


diabetes_agent = DiabetesAgent()


dspy.enable_logging()
diabetes_agent.agent.extract._compiled = True
diabetes_agent.agent.react._compiled = False

# Set up the teleprompter/optimizer using GEPA (per reference notebook)
teleprompter = GEPA(
    metric=llm_metric_prediction,
    max_full_evals=2,
    num_threads=32,
    track_stats=True,
    track_best_outputs=True,
    add_format_failure_as_feedback=True,
    reflection_lm=lm_teacher,
)

optimized_diabetes_agent = teleprompter.compile(student=diabetes_agent, trainset=trainset_diabetes, 
                                                valset=devset_diabetes)

# Access the detailed results from your optimized agent
results = optimized_diabetes_agent.detailed_results

# Get all candidates and their validation scores
candidates = results.candidates
val_scores = results.val_aggregate_scores

# Find the best candidate by validation score
best_idx = results.best_idx  # This is automatically calculated
best_score = val_scores[best_idx]
best_candidate = results.best_candidate

print(f"Best candidate index: {best_idx}")
print(f"Best validation score: {best_score}")
print(f"Best candidate components: {best_candidate}")

# save the optimized model in a new folder
os.makedirs("dspy_program", exist_ok=True)
optimized_diabetes_agent.save(
    "dspy_program/optimized_react_diabets.json", save_program=False
)


if __name__ == "__main__":
    # Test Snippet
    r = diabetes_agent(question="What are the main treatments for Type 2 diabetes?")
    print(r)




