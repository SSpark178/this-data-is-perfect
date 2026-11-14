import json
import os
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
INTEGER = re.compile(r"-?[0-9]+\Z")


def fields(record):
    result = {}
    for segment in record.split("|"):
        key, separator, value = segment.partition(":")
        key = key.strip()
        if separator and KEY.fullmatch(key):
            result[key] = value.strip()
    return result


def literal(raw):
    """Construiește vocabularul permis, fără asocierea cheie–valoare."""
    if raw == "":
        return "null"
    if raw in ("true", "false"):
        return raw
    if INTEGER.fullmatch(raw):
        return str(int(raw))
    return json.dumps(raw, ensure_ascii=True)


def terminal(text):
    return json.dumps(text, ensure_ascii=True)


def grammar_for(records):
    rules = [r"ws ::= [ \t\n\r]*"]

    for i, record in enumerate(records):
        fs = fields(record)
        vocabulary = list(dict.fromkeys(literal(v) for v in fs.values()))

        if vocabulary:
            rules.append(
                f"value{i} ::= "
                + " | ".join(terminal(v) for v in vocabulary)
            )

        members = [
            terminal(json.dumps(key)) + f' ws ":" ws value{i}'
            for key in fs
        ]
        body = ' ws "," ws '.join(members)
        rules.append(f'object{i} ::= "{{" ws {body} ws "}}"')

    objects = ' ws "," ws '.join(
        f"object{i}" for i in range(len(records))
    )
    return (
        f'root ::= ws "[" ws {objects} ws "]" ws\n'
        + "\n".join(rules)
    )


def violations(obj, record):
    fs = fields(record)
    if type(obj) is not dict or set(obj) != set(fs):
        return ["Missing or additional keys"]

    errors = []
    for key, raw in fs.items():
        value = obj[key]

        if raw == "":
            correct = value is None
        elif raw in ("true", "false"):
            correct = (
                type(value) is bool
                and value == (raw == "true")
            )
        elif INTEGER.fullmatch(raw):
            correct = type(value) is int and value == int(raw)
        else:
            correct = type(value) is str and value == raw

        if not correct:
            errors.append(
                f"Key {json.dumps(key)} must preserve input value "
                f"{json.dumps(raw)} using the conversion rules."
            )

    return errors


SYSTEM = """Convert OCR records to JSON.
Return an array, one object per input record.

Segments are separated by |.
Keep only segments whose text before the FIRST colon is a simple identifier.
Trim each key and value. Drop non-field scan noise.

Conversion:
- empty value -> null
- exact true or false -> boolean
- whole decimal integer with optional minus sign -> integer
- everything else -> string, copied exactly

Preserve quotes, backslashes and all punctuation inside values.
Do not remove annotations inside field values.
Use every real key once, in input order.
Never swap values between keys.
No explanations. /no_think"""


def prompt_for(records, feedback=""):
    user = json.dumps(records, ensure_ascii=True)

    if feedback:
        user += (
            "\nCorrect these errors from the previous attempt:\n"
            + feedback
        )

    return (
        "<|im_start|>system\n"
        + SYSTEM
        + "<|im_end|>\n"
        "<|im_start|>user\nOCR records (JSON strings):\n"
        + user
        + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def reject_constant(value):
    raise ValueError("Not strict JSON: " + value)


def decode_batch(llm, grammar_class, records):
    grammar = grammar_class.from_string(
        grammar_for(records), verbose=False
    )

    bound = 128
    for record in records:
        fs = fields(record)
        longest = max(
            (len(literal(v).encode("utf-8")) for v in fs.values()),
            default=0,
        )
        bound += sum(
            len(json.dumps(k).encode("utf-8")) + longest + 8
            for k in fs
        )

    feedback = ""

    for attempt in range(4):
        response = llm(
            prompt_for(records, feedback),
            grammar=grammar,
            max_tokens=bound,
            temperature=0.0 if attempt == 0 else 0.2,
            top_k=20,
            top_p=0.9,
            repeat_penalty=1.0,
            stop=["<|im_end|>", "<|endoftext|>"],
            echo=False,
        )

        try:
            objects = json.loads(
                response["choices"][0]["text"],
                parse_constant=reject_constant,
            )

            if type(objects) is not list or len(objects) != len(records):
                raise ValueError("Wrong number of objects")

            errors = []
            for i, (obj, record) in enumerate(zip(objects, records)):
                errors.extend(
                    f"Record {i + 1}: {error}"
                    for error in violations(obj, record)
                )

            if not errors:
                return objects

            feedback = "\n".join(errors)

        except (ValueError, TypeError) as error:
            feedback = str(error)

    raise RuntimeError(
        "Model failed validation; no hand-written fallback. "
        + feedback
    )


def main():
    from llama_cpp import Llama, LlamaGrammar

    model = BASE / "models" / "Qwen3-1.7B-Q8_0.gguf"
    if not model.is_file():
        raise FileNotFoundError(f"Missing local model: {model}")

    records = (BASE / "test-input.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    batch_size = int(os.environ.get("OCR_BATCH_SIZE", "4"))
    if batch_size < 1:
        raise ValueError("OCR_BATCH_SIZE must be positive")

    llm = Llama(
        model_path=str(model),
        n_gpu_layers=-1,
        n_ctx=8192,
        n_batch=2048,
        n_ubatch=512,
        flash_attn=True,
        n_threads=min(8, os.cpu_count() or 4),
        seed=42,
        verbose=False,
    )

    destination = BASE / "test-output.txt"
    temporary = BASE / "test-output.txt.tmp"

    try:
        with temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as output:
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]

                for obj in decode_batch(llm, LlamaGrammar, batch):
                    output.write(
                        json.dumps(
                            obj,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()