"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests.
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
import random

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""


def create_all_tasks():
    """Creates a dictionary of tasks from a list of levels
    :return: {task_name: task}
        e.g. {multiple-py: Task, multiple-java: Task}
    """
    return {"humaneval": create_task(True), "humaneval-unstripped": create_task(False)}


def create_task(strip_prompt):
    class HumanEval(GeneralHumanEval):
        def __init__(self, **kwargs):
            super().__init__(strip_prompt, **kwargs)

    return HumanEval


class GeneralHumanEval(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "openai_humaneval"

    def __init__(self, strip_prompt, k=[1, 10, 100], num_workers=16, timeout=3.0):
        super().__init__(
            stop_words=[
                "\nclass",
                "\ndef",
                "\n#",
                "\n@",
                "\nprint",
                "\nif",
                "\n```",
                "<file_sep>",
            ],
            requires_execution=True,
        )
        self.strip_prompt = strip_prompt
        # Parse pass@k values from arguments and validate against n_samples
        if hasattr(self.args, "pass_at_k"):
            k_values = [int(k) for k in self.args.pass_at_k.split(",")]
            n_samples = self.args.n_samples if hasattr(self.args, "n_samples") else 1
            # Filter out k values that are too large
            valid_k = [k for k in k_values if k <= n_samples]
            if len(valid_k) < len(k_values):
                print(
                    f"Warning: Removed k values > n_samples ({n_samples}). Using k={valid_k}"
                )
            self.k = valid_k if valid_k else [1]  # Default to [1] if all k were invalid
        else:
            self.k = k
        self.num_workers = num_workers
        self.timeout = timeout

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        if hasattr(self.args, "percent_problems") and self.args.percent_problems < 1.0:
            # Calculate number of problems to keep
            num_problems = len(dataset)
            num_to_keep = max(1, int(num_problems * self.args.percent_problems))

            if (
                hasattr(self.args, "sequential_problems")
                and self.args.sequential_problems
            ):
                # Take first n% of problems
                dataset = dataset.select(range(num_to_keep))
            else:
                # Randomly sample n% of problems
                indices = list(range(num_problems))
                random.seed(self.args.seed)  # Use same seed for reproducibility
                selected_indices = random.sample(indices, num_to_keep)
                dataset = dataset.select(selected_indices)
        return dataset

    def format_prompt(self, prompt_text):
        """Apply prompt template based on model type"""
        if (
            not hasattr(self.args, "prompt_template")
            or self.args.prompt_template == "plain"
        ):
            return prompt_text

        template = self.args.prompt_template.lower()
        if template == "wizardcoder":
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Create a Python script for this problem:
{prompt_text}

### Response:"""
        elif template == "codellama":
            return f"[INST] {prompt_text.strip()} [/INST]"
        elif template == "starcoder":
            return f"Question: {prompt_text}\n\nAnswer:\n"
        elif template == "octocoder":
            return f"Question: {prompt_text}\n\nAnswer:\n"
        elif template == "instructcodet5p":
            return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt_text}\n\n### Response:"
        elif template == "starchat":
            return (
                f"<|system|>\n<|end|>\n<|user|>\n{prompt_text}<|end|>\n<|assistant|>\n"
            )
        elif template == "deepseek":
            return f"You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{prompt_text.strip()}\n### Response:\n"
        elif template == "qwen":
            return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            print(f"Warning: Unknown prompt template '{template}'. Using plain format.")
            return prompt_text

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        if self.strip_prompt:
            prompt = doc["prompt"].strip()
        else:
            prompt = doc["prompt"]
        return self.format_prompt(prompt)

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        test_func = doc["test"]
        entry_point = f"check({doc['entry_point']})"
        return "\n" + test_func + "\n" + entry_point

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
            k=self.k,
            num_workers=self.num_workers,
            timeout=self.timeout,
        )
        return results
