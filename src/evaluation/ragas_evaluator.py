from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper


class RagasEvaluator:

    def __init__(self, llm):
        self.llm = LangchainLLMWrapper(llm)

    def evaluate(self, question, answer, chunks):
        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [chunks]
        })

        return evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ],
            llm=self.llm
        )