# data/

The dataset is **not** part of this repository: you bring your own skins.

```
data/
├── skins/          one 64×64 RGBA PNG per skin, named <id>.png
└── tags.json       one entry per skin (see tags.example.json)
```

`tags.json` is a list of objects:

| field   | meaning |
|---------|---------|
| `id`    | file name of the skin without `.png` |
| `tags`  | free-form tags, e.g. `"red hair"`, `"pirate"`, `"hoodie"` |
| `model` | arm model, `"classic"` (4 px arms) or `"slim"` (3 px arms) |

What `skingen/data.py` does with it:

- Skins that are not 64×64 (old 64×32 format) are skipped, exact duplicates (same pixels) too.
- Tags are lower-cased. Meaningless tags (`minecraft`, `skin`, …) and bare colours
  (`red`, `black`, …) are dropped, because a bare colour leaks into the whole skin.
  `red hair` is kept.
- Only tags that occur at least `MIN_TAG_COUNT` times (default 100) become part of the
  vocabulary.

Check the data before a long run:

```bash
python -m skingen.data
```

Paths can be changed in `.env` (see `.env.example`), so the data can live anywhere.
