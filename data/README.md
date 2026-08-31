# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Download `catalog.jsonl.gz` from the GitHub Release and decompress it as `catalog.jsonl` in this directory. Expected row count: 50,000.

Never place API keys, private evaluation data, or participant outputs in this directory.

## Integrity of the copy used for our reported results

The catalog is the organizer's frozen artifact and is not redistributed in this
repository. Download `catalog.jsonl.gz` from the participant kit release, verify
it against the published `SHA256SUMS`, decompress it, and place the result at
`data/catalog.jsonl`.

Every number we report was produced against a copy with this fingerprint:

```text
SHA256   da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67
bytes    60,546,327
records  50,000   (50,000 unique parent_asin, no duplicates)
fields   parent_asin, title, features, description, price, categories,
         details, average_rating, rating_number, store — present on every record
```

Confirm a local copy matches:

```bash
shasum -a 256 data/catalog.jsonl
wc -l data/catalog.jsonl
```

The catalog is read-only for us: nothing in `starter/` or `src/` opens it for
writing, and no ASIN is ever synthesized — recommendations can only contain
identifiers read from this file.
