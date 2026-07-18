from datasets import load_dataset
import json
from pathlib import Path

NUM_QUESTIONS = 10
OUTPUT_TEXT_PATH = "data/raw/hotpotqa_subset.txt"
OUTPUT_QUESTIONS_PATH = "data/eval/questions.json"


def main():
    print("Downloading HotpotQA (distractor, validation split)...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    selected = dataset.select(range(NUM_QUESTIONS))

    seen_titles = set()
    paragraph_blocks = []
    questions = []

    for example in selected:
        titles = example["context"]["title"]
        sentences_per_title = example["context"]["sentences"]

        for title, sentences in zip(titles, sentences_per_title):
            if title not in seen_titles:
                seen_titles.add(title)
                paragraph_text = " ".join(sentences)
                paragraph_blocks.append(f"== {title} ==\n{paragraph_text}")

        supporting_titles = example["supporting_facts"]["title"]
        supporting_sent_ids = example["supporting_facts"]["sent_id"]

        expected_facts = []
        for sup_title, sent_id in zip(supporting_titles, supporting_sent_ids):
            if sup_title in titles:
                title_index = titles.index(sup_title)
                sentence_list = sentences_per_title[title_index]
                if sent_id < len(sentence_list):
                    expected_facts.append(sentence_list[sent_id].strip())

        questions.append({
            "question": example["question"],
            "expected": expected_facts,
            "answer": example["answer"]
        })

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paragraph_blocks))

    Path("data/eval").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_QUESTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)

    print(f"Wrote {len(paragraph_blocks)} unique paragraphs to {OUTPUT_TEXT_PATH}")
    print(f"Wrote {len(questions)} questions to {OUTPUT_QUESTIONS_PATH}")


if __name__ == "__main__":
    main()
