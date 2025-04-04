import json
import math
import re
import warnings
from collections import defaultdict
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from bigcode_eval.criteria import EndOfFunctionCriteria, TooLongFunctionCriteria

INFILL_MODE = False
INSTRUCTION_MODE = False


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset
    Multiple copies of the same prompt are sent sequentially. See compute_code for more details.
    The prompt can either be:
    - one prompt: normal code completion
    - two prompts: for infilling mode (prefix, suffix) or instructin-tuning mode (instruction, context)
    """

    def __init__(
        self,
        task,
        dataset,
        tokenizer,
        num_devices,
        max_length,
        limit_start=0,
        n_tasks=None,
        n_copies=1,
        prefix="",
        has_encoder=False,
        instruction_tokens=None,
    ):
        self.task = task
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.num_devices = num_devices
        self.max_length = max_length
        self.limit_start = limit_start
        self.n_tasks = len(dataset) if n_tasks is None else n_tasks  # Use filtered dataset length
        self.n_copies = n_copies
        self.prefix = prefix
        self.has_encoder = has_encoder
        self.instruction_tokens = instruction_tokens

    def __iter__(self):
        prompts = []
        prompts_encoder = []
        infill = []
        instruction = []
        for sample in range(self.limit_start, self.limit_start + self.n_tasks):
            prompt_contents = self.task.get_prompt(self.dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                infill.append(False)
                instruction.append(False)
                prompt = self.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    # Infilling mode
                    infill.append(True)
                    instruction.append(False)
                    prompt = self._make_infill_prompt(
                        **prompt_contents, preprefix=self.prefix
                    )
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    instruction.append(True)
                    infill.append(False)
                    prompt = self._make_instruction_prompt(
                        **prompt_contents, prefix=self.prefix
                    )
            else:
                raise ValueError(f"Unsupported prompt format: {type(prompt_contents)}")
            prompts.append(prompt)
            if self.has_encoder:
                prompt_encoder = self.task.get_prompt_encoder(self.dataset[sample])
                if isinstance(prompt_encoder, str):
                    prompt_encoder = self.prefix + prompt_encoder
                prompts_encoder.append(prompt_encoder)

        if not len(set(infill)) == 1 or not len(set(instruction)) == 1:
            raise ValueError(
                "Mixed infill/instruction and completion prompts are not supported."
            )
        global INFILL_MODE
        global INSTRUCTION_MODE
        INFILL_MODE = infill[0]
        INSTRUCTION_MODE = instruction[0]
        if INFILL_MODE:
            return_token_type_ids = False
        else:
            return_token_type_ids = None  # default

        outputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length,
            return_token_type_ids=return_token_type_ids,
        )
        if self.has_encoder:
            outputs_encoder = self.tokenizer(
                prompts_encoder,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
                return_token_type_ids=return_token_type_ids,
            )

        if self.n_copies == 1 and self.n_tasks % self.num_devices != 0:
            self.n_copies = 2
            warnings.warn(
                "n_copies (n_samples/batch_size) was changed from 1 to 2 because n_tasks isn't proportional to num devices"
            )

        for sample in range(self.n_tasks):
            for _ in range(self.n_copies):
                if self.has_encoder:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "ids_encoder": outputs_encoder.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                        "input_len_encoder": outputs_encoder.attention_mask[
                            sample
                        ].sum(),
                    }
                else:
                    yield {
                        "ids": outputs.input_ids[sample],
                        "task_id": sample,
                        "input_len": outputs.attention_mask[sample].sum(),
                    }

    def _make_infill_prompt(self, prefix, suffix, preprefix=""):
        """Make a prompt for infilling.
        Currently supported only for official InCoder and SantaCoder implementations.
        """
        model_id = self.tokenizer.name_or_path
        if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
        elif model_id in ["bigcode/santacoder"]:
            return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
        elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
            return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
        else:
            raise ValueError(f"Infilling not yet supported for: {model_id}")

    def _make_instruction_prompt(self, instruction, context, prefix=""):
        """Make a prompt for instruction-tuning. Delimit instruction and context with specific tokens if provided."""
        if not self.instruction_tokens:
            warnings.warn(
                "Instruction-tuning tokens are not provided for an instruction-tuning task, we will leave them empty."
            )
            user_token, end_token, assistant_token = "", "", "\n"
        else:
            user_token, end_token, assistant_token = self.instruction_tokens
            if not user_token or not assistant_token or not end_token:
                warnings.warn(
                    "Instruction-tuning tokens provided but one or more are empty. Ignore warning if this was intended"
                )
        prompt = (
            prefix + user_token + instruction + end_token + assistant_token + context
        )

        return prompt


def _parse_infill(code, tokenizer):
    """Reorder infill code and remove remaining special tokens."""
    model_id = tokenizer.name_or_path
    if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
        prefix, suffix, infill = code.split("<|mask:0|>", 2)
        infill = infill.split("<|endofmask|>")[0]
    elif model_id in ["bigcode/santacoder"]:
        prefix, rest = code.split("<fim-suffix>", 1)
        suffix, infill = rest.split("<fim-middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
        prefix, rest = code.split("<fim_suffix>", 1)
        suffix, infill = rest.split("<fim_middle>", 1)
        infill = infill.split("<|endoftext|>")[0]
    else:
        raise ValueError(f"Infilling not yet supported for: {model_id}")
    for k, v in tokenizer.special_tokens_map.items():
        if k == "additional_special_tokens":
            for t in v:
                infill = infill.replace(t, "")
        else:
            infill = infill.replace(v, "")
    return infill


def _parse_instruction(code, instruction_tokens):
    """Return code block after assistant_token/end_token"""
    _, end_token, assistant_token = instruction_tokens
    if not assistant_token and end_token:
        assistant_token = end_token
    elif not assistant_token and not end_token:
        return code

    idx = code.find(assistant_token)
    shift = len(assistant_token)
    if idx == -1:
        warnings.warn(
            "The assistant token was not detected in the generation, this might disrupt the post-processing and lead to lower evaluation scores"
        )
        return code

    if "```python" in assistant_token:
        idx = code.find("```python", idx)
        shift = len("```python")
    return code[idx + shift :]


def complete_code(
    task,
    accelerator,
    model,
    tokenizer,
    dataloader,
    n_tasks,
    limit_start=0,
    batch_size=20,
    prefix="",
    instruction_tokens=None,
    postprocess=True,
    is_wrapped=False,
    save_every_k_tasks: int = -1,
    intermediate_generations: Optional[List[Optional[List[Optional[str]]]]] = None,
    intermediate_save_generations_path: Optional[str] = None,
    **gen_kwargs,
):
    """Complete code generations for a batch of tasks
    Args:
        task: task instance
        accelerator: accelerator instance
        model: model instance
        tokenizer: tokenizer instance
        dataloader: dataloader instance
        n_tasks: number of tasks to generate (after filtering)
        limit_start: index to start from
        batch_size: batch size
        prefix: prefix to add to each prompt
        instruction_tokens: tokens to use for instruction-tuning
        postprocess: whether to postprocess the generated code
        is_wrapped: whether the model is wrapped in accelerate
        save_every_k_tasks: save generations every k tasks
        intermediate_generations: list of intermediate generations
        intermediate_save_generations_path: path to save intermediate generations
        **gen_kwargs: additional arguments for model.generate
    """
    model = model if is_wrapped else accelerator.unwrap_model(model)
    all_input_ids = []
    all_input_ids_encoder = []
    all_task_ids = []
    all_input_len = []
    all_input_len_encoder = []

    # Collect all inputs first
    for batch in dataloader:
        task_ids = batch.pop("task_id")
        input_len = batch.pop("input_len")
        if "ids_encoder" in batch:
            input_len_encoder = batch.pop("input_len_encoder")
            all_input_len_encoder.extend(input_len_encoder.tolist())
            all_input_ids_encoder.extend(batch["ids_encoder"].tolist())
        all_input_ids.extend(batch["ids"].tolist())
        all_task_ids.extend(task_ids.tolist())
        all_input_len.extend(input_len.tolist())

    code_gens = defaultdict(list)
    n_samples = batch_size
    total_batch = math.ceil(len(all_task_ids) / n_samples)

    if accelerator.is_main_process:
        print(f"Processing {len(all_task_ids)} total inputs in {total_batch} batches")
        pbar = tqdm(total=total_batch, desc="Generating samples")
    else:
        pbar = None

    for i in range(0, len(all_task_ids), n_samples):
        # Get batch
        batch_input_ids = all_input_ids[i : i + n_samples]
        batch_task_ids = all_task_ids[i : i + n_samples]
        batch_input_len = all_input_len[i : i + n_samples]
        if all_input_ids_encoder:
            batch_input_ids_encoder = all_input_ids_encoder[i : i + n_samples]
            batch_input_len_encoder = all_input_len_encoder[i : i + n_samples]

        # Prepare model inputs
        batch_input_ids = torch.tensor(batch_input_ids).to(accelerator.device)
        if all_input_ids_encoder:
            batch_input_ids_encoder = torch.tensor(batch_input_ids_encoder).to(
                accelerator.device
            )

        # Update stopping criteria with actual input length
        if "stopping_criteria" in gen_kwargs:
            for criteria in gen_kwargs["stopping_criteria"]:
                if isinstance(criteria, EndOfFunctionCriteria):
                    criteria.start_length = batch_input_len[0]
                elif isinstance(criteria, TooLongFunctionCriteria):
                    criteria.input_length = batch_input_len[0]

        # Generate
        if all_input_ids_encoder:
            # seq2seq model
            outputs = model.generate(
                input_ids=batch_input_ids_encoder,
                decoder_input_ids=batch_input_ids,
                decoder_start_length=batch_input_len[0],
                **gen_kwargs,
            )
        else:
            # causal model
            outputs = model.generate(
                input_ids=batch_input_ids,
                **gen_kwargs,
            )

        # Put generations in a dict with task_id as key
        gen_dict = defaultdict(list)
        for task_id, output in zip(batch_task_ids, outputs.tolist()):
            gen_dict[task_id].append(output)

        # Update generations
        update_code_gens(
            task,
            tokenizer,
            limit_start,
            prefix,
            instruction_tokens,
            postprocess,
            code_gens,
            gen_dict,
        )

        if pbar is not None:
            pbar.update(1)

        # Save intermediate generations
        if (
            save_every_k_tasks > 0
            and (i + n_samples) % (save_every_k_tasks * n_samples) == 0
            and accelerator.is_main_process
        ):
            # Convert defaultdict to list
            generations = [[] for _ in range(n_tasks)]
            for task_id, gens in code_gens.items():
                generations[task_id] = gens
            # Add empty generations for remaining tasks
            if intermediate_generations:
                for task_id, gens in enumerate(intermediate_generations):
                    if gens:
                        generations[task_id] = gens
            with open(intermediate_save_generations_path, "w") as fp:
                json.dump(generations, fp)
                print(
                    f"intermediate generations saved at {intermediate_save_generations_path}"
                )

    if pbar is not None:
        pbar.close()

    # Convert defaultdict to list
    generations = [[] for _ in range(n_tasks)]
    for task_id, gens in code_gens.items():
        generations[task_id] = gens
    # Add empty generations for remaining tasks
    if intermediate_generations:
        for task_id, gens in enumerate(intermediate_generations):
            if gens:
                generations[task_id] = gens

    return generations


def update_code_gens(
    task,
    tokenizer,
    limit_start,
    prefix,
    instruction_tokens,
    postprocess,
    code_gens,
    gen_token_dict,
):  
    for sample, generated_tokens in gen_token_dict.items():
        for s in generated_tokens:
            if INFILL_MODE or tokenizer.eos_token in task.stop_words:
                if s[0] == tokenizer.bos_token_id:
                    s = s[1:]
                # Treat eos token as a regular stop word not removing it from the output
                # If it's removed it may have the effect of removing it in the middle of a
                # longer generation in case a batch size > 1 is used, which will result in
                # a wrong generation as it won't be used for splitting lateron
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                try:
                    # some tokenizers add a multi-token prefix to the generation (e.g ChatGLM)
                    tokenizer_prefix = tokenizer.decode(tokenizer.get_prefix_tokens())
                    if gen_code.startswith(f"{tokenizer_prefix}"):
                        gen_code = gen_code[len(tokenizer_prefix):].lstrip()
                except:
                    pass
                if INFILL_MODE:
                    gen_code = _parse_infill(gen_code, tokenizer)
                if INSTRUCTION_MODE:
                    gen_code = _parse_instruction(gen_code, instruction_tokens)
            else:
                gen_code = tokenizer.decode(
                    s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            if not INFILL_MODE:
                gen_code = gen_code[len(prefix) :]
            if postprocess:
                code_gens[sample].append(
                    task.postprocess_generation(gen_code, int(sample) + limit_start)
                )
            else:
                warnings.warn(
                    "model output is not postprocessed, this might lower evaluation scores"
                )
                code_gens[sample].append(gen_code)
    return code_gens


def remove_after_return(code):
    """
    Takes as input a code, and removes everything that is after the return.
    That is, the first line that does not start with a space character
    """
    pattern = r"[^\n]+(\n|$)"
    end_last_match = None
    # Go trough the regex to match any sequence of characters ending with a \n
    for match in re.finditer(pattern, code):
        start_match, end_match = match.span()
        # Search for the first line which does not start by a space character
        if (
            end_last_match is not None
            and start_match < len(code)
            and code[start_match].strip() != ""
        ):
            return code[0: start_match]
        end_last_match = end_match
    return code
