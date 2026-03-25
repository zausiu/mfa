"""Command line functions for aligning corpora"""
from __future__ import annotations

import typing
from pathlib import Path

import rich_click as click

from montreal_forced_aligner.alignment import PretrainedAligner 
from montreal_forced_aligner.command_line.utils import (
    common_options,
    initialize_configuration,
    validate_acoustic_model,
    validate_corpus_directory,
    validate_dictionary,
    validate_g2p_model,
)
from montreal_forced_aligner.data import WorkflowType

__all__ = ["align_corpus_cli"]


# commented by claude: This is the CLI entry point for `mfa align`. The Click framework
# decorators below define all command-line arguments and options that the user
# can pass. Each @click.argument / @click.option decorator adds one CLI param.
@click.command(
    name="align",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
        allow_interspersed_args=True,
    ),
    short_help="Align a corpus",
)
@click.argument("corpus_directory", type=click.UNPROCESSED, callback=validate_corpus_directory)
@click.argument("dictionary_path", type=click.UNPROCESSED, callback=validate_dictionary)
@click.argument("acoustic_model_path", type=click.UNPROCESSED, callback=validate_acoustic_model)
@click.argument(
    "output_directory", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    "--config_path",
    "-c",
    help="Path to config file to use for aligning.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--speaker_characters",
    "-s",
    help="Number of characters of file names to use for determining speaker, "
    "default is to use directory names.",
    type=str,
    default="0",
)
@click.option(
    "--audio_directory",
    "-a",
    help="Audio directory root to use for finding audio files.",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--reference_directory",
    help="Directory containing gold standard alignments to evaluate",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--custom_mapping_path",
    help="YAML file for mapping phones from acoustic model phone set to phone set in golden alignments.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output_format",
    help="Format for aligned output files (default is long_textgrid).",
    default="long_textgrid",
    type=click.Choice(["long_textgrid", "short_textgrid", "json", "csv"]),
)
@click.option(
    "--include_original_text",
    is_flag=True,
    help="Flag to include original utterance text in the output.",
    default=False,
)
@click.option(
    "--no_tokenization",
    is_flag=True,
    help="Flag to disable any pretrained tokenization.",
    default=False,
)
@click.option(
    "--fine_tune", is_flag=True, help="Flag for running extra fine tuning stage.", default=False
)
@click.option(
    "--fine_tune_boundary_tolerance",
    type=float,
    default=None,
    help="Flag for running extra fine tuning stage.",
)
@click.option(
    "--g2p_model_path",
    "g2p_model_path",
    help="Path to G2P model to use for OOV items.",
    type=click.UNPROCESSED,
)
@common_options
@click.help_option("-h", "--help")
@click.pass_context
def align_corpus_cli(context, **kwargs) -> None:
    """
    Align a corpus with a pronunciation dictionary and a pretrained acoustic model.
    """
    # commented by claude: Merge global MFA config with any user-provided config file.
    # initialize_configuration reads ~/.config/mfa or the --config_path file and
    # sets global state (e.g. number of jobs, temp directory) before anything else.
    initialize_configuration(context)
    config_path = kwargs.get("config_path", None)
    reference_directory: typing.Optional[Path] = kwargs.get("reference_directory", None)
    custom_mapping_path: typing.Optional[Path] = kwargs.get("custom_mapping_path", None)
    corpus_directory = kwargs["corpus_directory"].absolute()
    dictionary_path = kwargs["dictionary_path"]
    acoustic_model_path = kwargs["acoustic_model_path"]
    output_directory = kwargs["output_directory"]
    output_format = kwargs["output_format"]
    include_original_text = kwargs["include_original_text"]

    # commented by claude: parse_parameters merges the YAML config file (if given via -c)
    # with the CLI arguments into a single dict that will be passed as **kwargs
    # to PretrainedAligner. This is the central place where config priority is resolved.
    extra_kwargs = PretrainedAligner.parse_parameters(config_path, context.params, context.args)
    no_tokenization = kwargs["no_tokenization"]
    if no_tokenization:
        # commented by claude: Setting language="unknown" disables any language-specific
        # tokenizer/normalizer that would otherwise process the text before alignment.
        extra_kwargs["language"] = "unknown"
    g2p_model_path: typing.Optional[Path] = kwargs.get("g2p_model_path", None)
    if g2p_model_path:
        g2p_model_path = validate_g2p_model(context, kwargs, g2p_model_path)

    # commented by claude: Instantiate the top-level aligner object. At this point no
    # heavy work happens yet — the acoustic model file is opened but the corpus is
    # NOT loaded and the database is NOT created. That happens inside aligner.align().
    aligner = PretrainedAligner(
        corpus_directory=corpus_directory,
        dictionary_path=dictionary_path,
        acoustic_model_path=acoustic_model_path,
        g2p_model_path=g2p_model_path,
        **extra_kwargs,
    )
    try:
        # commented by claude: STEP 1 — Core alignment pipeline.
        # Internally calls setup() (DB init, corpus load, acoustic model export),
        # then compile_train_graphs(), _align() (Kaldi GMM), and collect_alignments()
        # which writes phone/word intervals into the SQLite/PostgreSQL database.
        aligner.align()

        # commented by claude: STEP 2 — Quality analysis pass.
        # For every aligned utterance, computes: speech log-likelihood, duration
        # deviation (Z-score vs expected phone durations), and SNR. Results are written
        # back to the `utterance` table and exported as alignment_analysis.csv.
        aligner.analyze_alignments()

        # commented by claude: STEP 3 — Export aligned intervals to output files.
        # Reads phone/word intervals from the database and writes TextGrid (or JSON/CSV)
        # files. One output file is created per input sound file.
        aligner.export_files(
            output_directory,
            output_format=output_format,
            include_original_text=include_original_text,
        )

        # commented by claude: OPTIONAL STEP 4 — Evaluation against gold-standard alignments.
        # Only runs if the user passed --reference_directory or if reference alignments
        # were previously stored in the database. Computes phone error rate (PER) and
        # other metrics comparing the automatic alignment to the reference.
        if reference_directory or aligner.has_reference_alignments():
            if reference_directory:
                aligner.load_reference_alignments(reference_directory)
            else:
                aligner.check_manual_alignments()

            if custom_mapping_path is not None:
                aligner.load_mapping(custom_mapping_path)
            reference_alignments = WorkflowType.reference
        else:
            reference_alignments = WorkflowType.alignment

        if reference_alignments is WorkflowType.reference:
            aligner.evaluate_alignments(
                output_directory=output_directory,
                reference_source=reference_alignments,
                comparison_source=WorkflowType.alignment,
            )
    except Exception:
        # commented by claude: Mark the aligner as "dirty" so that on the next run MFA
        # knows the previous run did not complete cleanly and may prompt the user
        # to re-run with --clean to avoid stale partial results.
        aligner.dirty = True
        raise
    finally:
        # commented by claude: Always runs, even after an exception. Closes DB connections,
        # cleans up temp files (if configured), and releases resources.
        aligner.cleanup()
