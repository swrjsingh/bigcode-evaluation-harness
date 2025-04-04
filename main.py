import os
import fnmatch
import json
import warnings
import logging

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
)

from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import ALL_TASKS
from bigcode_eval import tasks


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)

    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=-1,
        help="Optional saving after every k tasks",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        help="List of paths for saving the intermediate code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def load_hf_lm_and_tokenizer(
    model_name,
    tokenizer_name=None,
    load_in_8bit=False,
    load_in_4bit=False,
    device_map=None,
    torch_dtype=None,
    revision=None,
    use_auth_token=None,
    trust_remote_code=False,
):
    """Load HuggingFace model and tokenizer."""
    if tokenizer_name is None:
        tokenizer_name = model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=revision,
        trust_remote_code=trust_remote_code,
        use_auth_token=use_auth_token,
        truncation_side="left",
        padding_side="right",
    )

    # Set padding token if not set
    if not tokenizer.pad_token:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
            print("Using eos_token as pad_token")
        else:
            raise ValueError("No pad_token or eos_token found")

    # Load the model
    if load_in_8bit:
        print("Loading model in 8bit")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map=device_map,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
    elif load_in_4bit:
        print("Loading model in 4bit")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map=device_map,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
    else:
        print(f"Loading model in {torch_dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )

    return tokenizer, model


def main():
    args = parse_args()
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")
        if hasattr(args, "percent_problems"):
            print(f"Using {args.percent_problems*100}% of problems")
            if hasattr(args, "sequential_problems") and args.sequential_problems:
                print("Selecting problems sequentially from start")
            else:
                print(f"Randomly selecting problems with seed {args.seed if hasattr(args, 'seed') else 'not set'}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.precision == "bf16":
        print("Loading model in bf16")
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        print("Loading model in fp16")
        torch_dtype = torch.float16
    else:
        print("Loading model in fp32")
        torch_dtype = torch.float32

    # Load tokenizer and model
    tokenizer, model = load_hf_lm_and_tokenizer(
        model_name=args.model,
        tokenizer_name=args.model,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=torch_dtype,
        device_map="auto" if args.precision != "fp32" else None,
        revision=args.revision,
        use_auth_token=args.use_auth_token,
        trust_remote_code=args.trust_remote_code,
    )

    results = {}
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        args=args,
    )

    for task in task_names:
        results[task] = evaluator.evaluate(task)

    # Save all args to config
    results["config"] = vars(args)
    if not args.generation_only:
        dumped = json.dumps(results, indent=2)
        if accelerator.is_main_process:
            print(dumped)

        with open(args.metric_output_path, "w") as f:
            f.write(dumped)


if __name__ == "__main__":
    main()
