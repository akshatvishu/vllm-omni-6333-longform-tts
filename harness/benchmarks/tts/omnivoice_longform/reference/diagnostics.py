# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import soundfile as sf


def _task_record(task, frame_rate):
    return {
        "texts": list(task.texts),
        "target_tokens": [int(value) for value in task.target_lens],
        "estimated_duration_s": [int(value) / frame_rate for value in task.target_lens],
        "reference_texts": list(task.ref_texts),
        "reference_tokens": [int(value.shape[-1]) if value is not None else 0 for value in task.ref_audio_tokens],
    }


@contextmanager
def capture_generation(model, enabled):
    """Observe one instance; retain decoded arrays without additional generation."""
    record = {"chunks": [], "decoded": [], "outputs": {}}
    frame_rate = model.audio_tokenizer.config.frame_rate
    originals = {}

    def patch(obj, name, method):
        # Restore inherited methods by removing the temporary instance override.
        originals[(obj, name)] = (name in vars(obj), vars(obj).get(name))
        setattr(obj, name, method)

    preprocess = model._preprocess_all
    iterative = model._generate_iterative

    def preprocess_wrapper(*args, **kwargs):
        task = preprocess(*args, **kwargs)
        record["request"] = _task_record(task, frame_rate)
        return task

    def iterative_wrapper(task, gen_config):
        record["chunks"].append(_task_record(task, frame_rate))
        return iterative(task, gen_config)

    patch(model, "_preprocess_all", preprocess_wrapper)
    patch(model, "_generate_iterative", iterative_wrapper)
    if enabled:
        decode = model.audio_tokenizer.decode
        postprocess = model._post_process_audio

        def decode_wrapper(*args, **kwargs):
            result = decode(*args, **kwargs)
            record["decoded"].append(result.audio_values[0].detach().cpu().numpy().copy())
            return result

        def postprocess_wrapper(generated_audio, ref_rms, gen_config):
            record["outputs"]["joined_before_postprocess"] = generated_audio.copy()
            record["outputs"]["silence_removal_disabled"] = postprocess(
                generated_audio.copy(), ref_rms, replace(gen_config, postprocess_output=False)
            )
            return postprocess(generated_audio, ref_rms, gen_config)

        patch(model.audio_tokenizer, "decode", decode_wrapper)
        patch(model, "_post_process_audio", postprocess_wrapper)
    try:
        yield record
    finally:
        for (obj, name), (was_local, original) in originals.items():
            if was_local:
                setattr(obj, name, original)
            else:
                delattr(obj, name)


def save_capture(record, output_dir, case_id, sample_rate):
    decoded = record.pop("decoded")
    outputs = record.pop("outputs")
    record["chunk_count"] = sum(len(item["texts"]) for item in record["chunks"])
    record["chunk_audio"] = []
    for index, audio in enumerate(decoded):
        audio = np.asarray(audio).squeeze()
        path = output_dir / f"{case_id}.chunk{index:03d}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        record["chunk_audio"].append(
            {
                "path": str(path.resolve()),
                "duration_s": audio.size / sample_rate,
                "rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))),
                "peak": float(np.max(np.abs(audio))),
            }
        )
    record["output_variants"] = {}
    for name, audio in outputs.items():
        path = output_dir / f"{case_id}.{name}.wav"
        sf.write(path, np.asarray(audio).squeeze(), sample_rate, subtype="FLOAT")
        record["output_variants"][name] = str(path.resolve())
    return record
