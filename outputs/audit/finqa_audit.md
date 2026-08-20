# FinQA audit

- Seed: 42

## train

- Records: 6251
- Duplicate IDs: 0
- Invalid tables: 0
- SHA256: `49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6`

## dev

- Records: 883
- Duplicate IDs: 0
- Invalid tables: 0
- SHA256: `a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51`

## test

- Records: 1147
- Duplicate IDs: 0
- Invalid tables: 0
- SHA256: `831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc`

## Cross-split overlap

```json
{
  "id": {
    "train__dev": 0,
    "train__test": 0,
    "dev__test": 0
  },
  "report_page": {
    "train__dev": 0,
    "train__test": 0,
    "dev__test": 0
  },
  "company": {
    "train__dev": 96,
    "train__test": 99,
    "dev__test": 78
  }
}
```
