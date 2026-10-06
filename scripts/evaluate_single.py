from benchmarks.tts.omnivoice_longform import evaluate


def validate_single_backend(rows):
    assert rows and {row["backend"] for row in rows} == {"vllm-omni"}
    assert len({row["case_id"] for row in rows}) == len(rows)


evaluate._validate_backend_cases = validate_single_backend
evaluate.run(evaluate.parse_args())
