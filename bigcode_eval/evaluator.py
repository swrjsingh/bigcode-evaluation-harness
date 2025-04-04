import inspect
import json
import os
import warnings

from typing import List

from tqdm import tqdm

from bigcode_eval import tasks
from bigcode_eval.generation import parallel_generations

_WARNING = """
################################################################################
                                  !!!WARNING!!!
################################################################################
The "code_eval"/"apps_metric" you are about to use, execute untrusted 
model-generated code in Python.
Although it is highly unlikely that model-generated code will do something
overtly malicious in response to this test suite, model-generated code may act
destructively due to a lack of model capability or alignment.
Users are strongly encouraged to sandbox this evaluation suite so that it
does not perform destructive actions on their host or network. For more
information on how OpenAI sandboxes its code, see the paper "Evaluating Large
Language Models Trained on Code" (https://arxiv.org/abs/2107.03374).
Once you have read this disclaimer and taken appropriate precautions, set the argument 
"allow_code_execution" to True.
################################################################################\
"""

class Evaluator:
    def __init__(self, accelerator, model, tokenizer, args):
        self.accelerator = accelerator
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        # setup arguments
        self.metric_output_path = args.metric_output_path

        # code evaluation permission
        self.allow_code_execution = args.allow_code_execution

    def generate_text(self, task_name, total_samples=None, intermediate_generations=None):
        """Generates text for task
        :param task_name: str
            name of task to evaluate on
        :param total_samples: int
            total number of samples to generate (problems * n_samples)
        :param intermediate_generations: list[list[str | None]]
            list of lists of generated codes or empty
        :return: tuple[list[list[str]], list[str]]
            tuple of (generations, references)
        """
        task = tasks.get_task(task_name, self.args)
        dataset = task.get_dataset()  # This now handles sequential/random selection
        n_tasks = len(dataset)  # This is now the filtered size
        n_samples = self.args.n_samples if hasattr(self.args, "n_samples") else 1
        total_samples = n_tasks * n_samples

        if self.accelerator.is_main_process:
            print(f"Generating {n_samples} samples for each of {n_tasks} problems (total {total_samples} samples)...")

        generations = []
        references = [task.get_reference(dataset[i]) for i in range(n_tasks)]

        if self.args.check_references:
            if "get_solution" in inspect.signature(task.get_reference).parameters:
                solutions = [[task.get_reference(dataset[i], get_solution=True)] for i in range(n_tasks)]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        curr_generations = []
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)
        intermediate_save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task_name}_intermediate.json"
        curr_sample_idx = len(curr_generations)

        generations = parallel_generations(
            task,
            dataset,
            self.accelerator,
            self.model,
            self.tokenizer,
            n_tasks=n_tasks,
            args=self.args,
            curr_sample_idx=curr_sample_idx,
            save_every_k_tasks=self.args.save_every_k_tasks,
            intermediate_generations=curr_generations,
            intermediate_save_generations_path=intermediate_save_generations_path,
        )

        if len(generations[0]) > n_samples:
            generations = [l[:n_samples] for l in generations]
            warnings.warn(
                f"Number of tasks wasn't proportional to number of devices, we removed extra predictions to only keep nsamples={n_samples}"
            )
        return generations, references

    def evaluate(self, task, total_samples=None, intermediate_generations=None):
        """Evaluates the model on a task from the benchmark
        :param task: str
            name of task to evaluate on
        :param total_samples: int
            total number of samples to generate (problems * n_samples)
        :param intermediate_generations: list[list[str | None]]
            list of lists of generated codes or empty
        :return: dict[str: float]
            dict containing evaluation metrics
        """
        task_obj = tasks.get_task(task, self.args)
        
        if self.args.load_generations_path:
            generations = self.load_generations(task)
            references = self.load_references(task)
            return task_obj.process_results(generations, references)

        if self.args.check_references:
            solutions, references = self.generate_text(task, intermediate_generations=intermediate_generations)
            return {
                "solutions": solutions,
                "references": references
            }

        # Get filtered dataset size
        dataset = task_obj.get_dataset()
        n_tasks = len(dataset)
        n_samples = self.args.n_samples if hasattr(self.args, "n_samples") else 1
        total_samples = n_tasks * n_samples

        generations, references = self.generate_text(
            task, total_samples=total_samples, intermediate_generations=intermediate_generations
        )

        if self.accelerator.is_main_process:
            if not self.args.load_generations_path:
                # Create directory if it doesn't exist
                save_generations_path = f"{os.path.splitext(self.args.save_generations_path)[0]}_{task}.json"
                os.makedirs(os.path.dirname(save_generations_path), exist_ok=True)
                self.save_json_files(generations, references, save_generations_path, f"references_{task}.json")

            # make sure tokenizer plays nice with multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if self.allow_code_execution and task_obj.requires_execution:
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            print("Evaluating generations...")
            results = task_obj.process_results(generations, references)
            return results

    def save_json_files(
        self,
        generations: List[str],
        references: List[str],
        save_generations_path: str,
        save_references_path: str,
    ) -> None:
        if self.args.save_generations:
            with open(save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(f"generations were saved at {save_generations_path}")
        if self.args.save_references:
            with open(save_references_path, "w") as fp:
                json.dump(references, fp)
                print(f"references were saved at {save_references_path}")
