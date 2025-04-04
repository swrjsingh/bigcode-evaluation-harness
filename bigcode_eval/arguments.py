from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArguments:
    """
    Configuration for running the evaluation.
    """

    prefix: Optional[str] = field(
        default="",
        metadata={
            "help": "Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"
        },
    )
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Sample from the language model's output distribution."},
    )
    temperature: Optional[float] = field(
        default=0.2, metadata={"help": "Sampling temperature used for generation."}
    )
    top_k: Optional[int] = field(
        default=0, metadata={"help": "Top-k parameter used for generation."}
    )
    top_p: Optional[float] = field(
        default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."}
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help": "Number of completions to generate for each sample."},
    )
    eos: Optional[str] = field(
        default="<|endoftext|>", metadata={"help": "end of sentence token."}
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed used for evaluation."}
    )
    percent_problems: Optional[float] = field(
        default=1.0,
        metadata={"help": "Percentage of problems to evaluate (between 0 and 1)"},
    )
    sequential_problems: Optional[bool] = field(
        default=True,
        metadata={
            "help": "If True, take first n% of problems. If False, randomly sample n%"
        },
    )
    pass_at_k: Optional[str] = field(
        default="1,5,10",
        metadata={
            "help": "Comma-separated list of k values for pass@k computation. Each k must be <= n_samples"
        },
    )
    prompt_template: Optional[str] = field(
        default="plain",
        metadata={
            "help": "Prompt template to use. Options: plain, wizardcoder, codellama, starcoder, octocoder, instructcodet5p, etc."
        },
    )
