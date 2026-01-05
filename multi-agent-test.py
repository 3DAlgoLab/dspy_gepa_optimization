import dspy
import os
from dotenv import load_dotenv

import json
import logging
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
import random



import mlflow
from common import LM_MODEL_STUDENT, LM_MODEL_TEACHER

# Note: mlflow.dspy.autolog() may not be available in all versions
# Uncomment the following lines if your mlflow version supports dspy integration
try:
    mlflow.dspy.autolog(
        log_compiles=True,  # Track optimization process
        log_evals=True,  # Track evaluation results
        log_traces_from_compile=True,  # Track program traces during optimization
    )
    mlflow.dspy.autolog()
except AttributeError:
    print(
        "Warning: mlflow.dspy.autolog() not available, proceeding without MLflow logging"
    )
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("med-ai-workshop-test")

load_dotenv()

# Configure basic logging if not already configured by the host app
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("metric")




"""
Build a FAISS vector DB from two local PDF papers using LangChain.
Outputs a directory "faiss_index" with the persisted vectorstore.
"""




# Configure your LM (DSPy tutorial uses dspy.LM)
lm = dspy.LM(
    LM_MODEL_STUDENT,
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=64000,
)
dspy.settings.configure(lm=lm)

# Teacher LM for reflection (GEPA)
lm_teacher = dspy.LM(
    LM_MODEL_TEACHER,
    model_type="chat",
    cache=False,
    temperature=0.3,
    max_tokens=64000,
)





# Define the metric for evaluation using an LLM to check for factual consistency.
class JudgeConsistency(dspy.Signature):
    """Judge whether the predicted answer matches the gold answer.

    # Instructions:
    - The score should be between 0.0 and 1.0 and based on the similarity of the predicted answer and the gold answer.
    - The justification should be a brief explanation of the score.
    - If the answer doesn't address the question properly, the score should be less than 0.5.
    - If the answer is completely correct, the score should be 1.0. Otherwise, the score should be less than 1.0.
    - Be very strict in your judgement as this is a medical question.
    """

    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    justification: str = dspy.OutputField()


class JudgeReactStep(dspy.Signature):
    """Judge whether the next tool call (name + args) is appropriate, well-formed, and relevant.

    - Output a strict score in [0, 1].
    - Provide a brief justification and a yes/no style verdict in justification text.
    """

    question: str = dspy.InputField()
    tool_name: str = dspy.InputField()
    tool_args_json: str = dspy.InputField()
    score: float = dspy.OutputField(desc="a float score between 0 and 1")
    verdict: str = dspy.OutputField()
    justification: str = dspy.OutputField()


def llm_metric_prediction(*args, **kwargs):
    """Metric returning ScoreWithFeedback for GEPA and Evaluate.

    Accepts flexible arguments because GEPA may pass additional positional
    parameters (e.g., program, trace, batch metadata). We only need `example`
    and `pred` here; the rest are ignored.
    """
    # GEPA may pass predictor context for per-predictor feedback
    pred_name = kwargs.get("pred_name")
    pred_trace = kwargs.get("pred_trace")
    if pred_name is not None:
        logger.info(f"metric called for predictor={pred_name}")

    # Extract example and prediction from positional/keyword args
    example = kwargs.get("example") or kwargs.get("gold")
    pred = kwargs.get("pred") or kwargs.get("prediction")
    if example is None and len(args) > 0:
        example = args[0]
    if pred is None and len(args) > 1:
        pred = args[1]

    # Special handling: when optimizing the ReAct loop predictor
    if (
        pred_name
        and (pred_name == "react" or pred_name.endswith(".react"))
        and pred_trace
    ):
        try:
            _, step_inputs, step_outputs = pred_trace[0]
        except Exception:
            step_inputs, step_outputs = {}, {}

        question_text = (
            getattr(example, "question", None) or step_inputs.get("question", "") or ""
        )

        # Read tool name/args from the predictor's outputs (dict or Prediction)
        def _get(o, key, default=""):
            if isinstance(o, dict):
                return o.get(key, default)
            return getattr(o, key, default)

        tool_name = _get(step_outputs, "next_tool_name", "")
        tool_args = _get(step_outputs, "next_tool_args", {})

        # Heuristics: well-formed JSON args and sensible fields
        args_is_dict = isinstance(tool_args, dict)
        has_query = (
            args_is_dict
            and isinstance(tool_args.get("query"), str)
            and tool_args.get("query", "").strip() != ""
        )
        k_val = tool_args.get("k") if args_is_dict else None
        k_ok = isinstance(k_val, int) and 1 <= k_val <= 10 or k_val is None
        used_tool = tool_name not in ("", "finish")
        early_finish = tool_name == "finish"

        logger.debug(
            "react-step details | used_tool=%s tool=%s args_keys=%s has_query=%s k=%s early_finish=%s pred_trace_len=%s",
            used_tool,
            str(tool_name),
            (
                list(tool_args.keys())
                if isinstance(tool_args, dict)
                else type(tool_args).__name__
            ),
            has_query,
            k_val,
            early_finish,
            len(pred_trace) if pred_trace else 0,
        )

        heuristics_score = 0.0
        if used_tool:
            heuristics_score += 0.4
        if has_query:
            heuristics_score += 0.4
        if k_ok:
            heuristics_score += 0.1
        if not early_finish:
            heuristics_score += 0.1
        heuristics_score = max(0.0, min(1.0, heuristics_score))

        # LLM judge for the loop step (tool choice + query relevance)
        tool_args_json = (
            json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
        )
        with dspy.settings.context(lm=lm):
            react_judge = dspy.Predict(JudgeReactStep)
            judged = react_judge(
                question=question_text,
                tool_name=str(tool_name),
                tool_args_json=tool_args_json,
            )

        llm_score = getattr(judged, "score", 0.0) or 0.0
        llm_score = max(0.0, min(1.0, llm_score))
        llm_just = getattr(judged, "justification", "") or ""

        total = 0.5 * heuristics_score + 0.5 * llm_score

        logger.info(
            "react-step scores | heuristics=%.3f llm=%.3f total=%.3f",
            heuristics_score,
            llm_score,
            total,
        )

        # Actionable feedback
        suggestions = []
        if not used_tool:
            suggestions.append("Select a retrieval tool before finishing.")
        if early_finish:
            suggestions.append(
                "Avoid selecting 'finish' until you have evidence from the retrieval tool."
            )
        if not args_is_dict:
            suggestions.append("Emit next_tool_args as a valid JSON object.")
        else:
            if not has_query:
                suggestions.append(
                    "Include a non-empty 'query' string in next_tool_args."
                )
            if k_val is not None and (
                not isinstance(k_val, int) or k_val < 1 or k_val > 10
            ):
                suggestions.append("Choose a reasonable k (e.g., 3–5).")
        if not suggestions:
            suggestions.append(
                "Good step. Keep queries concise and set k=5 by default."
            )

        feedback_text = (
            f"ReAct step — LLM score: {llm_score:.2f}, heuristics: {heuristics_score:.2f}. "
            + " ".join(suggestions)
            + (f" LLM justification: {llm_just}" if llm_just else "")
        ).strip()

        return ScoreWithFeedback(score=total, feedback=feedback_text)

    # Program-level or non-react predictor: judge final answer quality
    # Defensive checks
    if example is None or pred is None:
        return ScoreWithFeedback(score=0.0, feedback="Missing example or pred")

    predicted_answer = getattr(pred, "answer", None) or ""
    if not predicted_answer.strip():
        return ScoreWithFeedback(score=0.0, feedback="Empty prediction")

    with dspy.settings.context(lm=lm):
        judge = dspy.Predict(JudgeConsistency)
        judged = judge(
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=predicted_answer,
        )

    score = getattr(judged, "score", None) or 0.0
    score = max(0.0, min(1.0, score))
    justification = getattr(judged, "justification", "") or ""
    logger.info(
        "answer-level score=%.3f for question='%s'",
        score,
        (
            (example.question[:80] + "...")
            if len(example.question) > 80
            else example.question
        ),
    )
    feedback_text = f"Score: {score}. {justification}".strip()
    return ScoreWithFeedback(score=score, feedback=feedback_text)


def do_optimize():
    print("-" * 80)
    print("Start Optimizaing...")
    print("-" * 80)

    # Load the dataset
    with open("docs/qa_pairs_diabets.json", "r") as f:
        qa_diabetes_data = json.load(f)

    # Convert to dspy.Example objects
    dataset_diabetes = [
        dspy.Example(question=item["question"], answer=item["answer"]).with_inputs(
            "question"
        )
        for item in qa_diabetes_data
    ]

    # shuffle the dataset
    random.shuffle(dataset_diabetes)

    # Split the dataset as requested
    train_size = 8
    trainset_diabetes = dataset_diabetes[:train_size]
    devset_diabetes = dataset_diabetes[train_size:]

    print("Diabetes Dataset Ready...")
    print(f"Loaded {len(dataset_diabetes)} examples.")
    print(f"Train set size: {len(trainset_diabetes)}")
    print(f"Dev set size: {len(devset_diabetes)}")

    # Load the dataset
    with open("docs/qa_pairs_copd.json", "r") as f:
        qa_copd_data = json.load(f)

    # Convert to dspy.Example objects
    dataset_copd = [
        dspy.Example(question=item["question"], answer=item["answer"]).with_inputs(
            "question"
        )
        for item in qa_copd_data
    ]

    # shuffle the dataset
    random.shuffle(dataset_copd)

    # Split the dataset as requested
    trainset_copd = dataset_copd[:10]
    devset_copd = dataset_copd[10:]

    print("COPD Dataset Ready...")
    print(f"Loaded {len(dataset_copd)} examples.")
    print(f"Train set size: {len(trainset_copd)}")
    print(f"Dev set size: {len(devset_copd)}")

    # Prepare evaluation
    evaluator_diabetes = Evaluate(
        devset=devset_diabetes,
        num_threads=32,
        display_progress=True,
        display_table=5,
        provide_traceback=True,
    )


if __name__ == "__main__":
    test_tools()
    # rag_test()
    do_optimize()
