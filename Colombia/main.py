"""python main.py --data reviews.json --output result
Uses the trusted reviews to validate the public source corpus, then compares
suspect reviews with that corpus and opens every seal as instructed by the note.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import urllib.request

SOURCE = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Musical_Instruments_5.json.gz'
PUNCTUATION = str.maketrans({'’': "'", '–': '-'})
REPAIR = str.maketrans({'\u200b': '', 'а': 'a', 'с': 'c', 'е': 'e', 'о': 'o'})


def canonical(summary, text):
    return summary.translate(PUNCTUATION).translate(REPAIR), text.translate(PUNCTUATION).translate(REPAIR)


def solve(data_path, source_path, output):
    data = json.loads(data_path.read_text())
    if not source_path.exists():
        urllib.request.urlretrieve(SOURCE, source_path)
    lookup = {}
    with gzip.open(source_path, 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            key = canonical(r.get('summary', ''), r.get('reviewText', ''))
            lookup.setdefault(key, set()).add(int(r['overall']))
    # Validate this source version against every supplied trusted reference.
    for r in data['trusted_reference']:
        key = canonical(r['summary'], r['review_text'])
        if key not in lookup or r['rating'] not in lookup[key]:
            raise ValueError('Public source does not agree with trusted reference')
    allowed = {c for r in data['trusted_reference'] for f in ['summary', 'review_text'] for c in r[f]}
    altered = []
    for r in data['samples']:
        key = canonical(r['summary'], r['review_text'])
        if key not in lookup or len(lookup[key]) != 1:
            raise ValueError(f'Missing or ambiguous original review: {r["sample_index"]}')
        original_rating = next(iter(lookup[key]))
        unusual = sorted({f'U+{ord(c):04X}' for f in ['summary', 'review_text'] for c in r[f] if c not in allowed})
        reasons = []
        if unusual:
            reasons.append('unicode:' + '+'.join(unusual))
        if r['rating'] != original_rating:
            reasons.append('rating')
        if reasons:
            altered.append(dict(sample_index=r['sample_index'], reason=';'.join(reasons),
                                supplied_rating=r['rating'], original_rating=original_rating))
    altered.sort(key=lambda r: r['sample_index'])
    indices = [r['sample_index'] for r in altered]
    payload = ','.join(map(str, indices))
    mask = hashlib.sha256(payload.encode('utf-8')).digest()
    messages = []
    for group in ['trusted_reference', 'samples']:
        for i, r in enumerate(data[group]):
            encrypted = bytes.fromhex(r['archive_tag'])
            if len(encrypted) != len(mask):
                raise ValueError('Unexpected seal length')
            clear = bytes(a ^ b for a, b in zip(encrypted, mask)).rstrip(b'\0')
            if re.fullmatch(rb'SIGMOID_[A-Z_]+', clear):
                messages.append(dict(flag=clear.decode('ascii'), group=group, row_index=i))
    if len(messages) != 1:
        raise ValueError(f'Expected one recovered flag, found {len(messages)}')
    output.mkdir(parents=True, exist_ok=True)
    result = messages[0] | dict(altered_indices=indices, mask_sha256=mask.hex(),
        validated_trusted_reviews=len(data['trusted_reference']), source_url=SOURCE,
        source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest())
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'flag.txt').write_text(result['flag']+'\n')
    with (output/'altered_reviews.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_index','reason','supplied_rating','original_rating'])
        writer.writeheader()
        writer.writerows(altered)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('original.json.gz'))
    parser.add_argument('--output', type=Path, default=Path('result'))
    args = parser.parse_args()
    solve(args.data, args.source, args.output)
