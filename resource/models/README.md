# Models-page snapshot

The captured page lists 4 model instances. Metadata, descriptions, and available
model cards are saved. Required Gemma v2 weights were verified separately; other weights were not downloaded.
At capture time, official scoring required the quantized Gemma 31B base; later host clarifications
are recorded in docs/competition-research.md.
Other entries on this page do not imply approval for submissions.

| Instance | Version | Package bytes | Local card / instance record |
| --- | ---: | ---: | --- |
| [google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct](https://www.kaggle.com/models/google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct) | 2 | 23297590856 | [Card](google/gemma-4/model.json) / [Instance](google/gemma-4/instance.json) |
| [jek1wantaufik/buddy/Transformers/gemma4](https://www.kaggle.com/models/jek1wantaufik/buddy/Transformers/gemma4) | 1 | 8989 | [Card](jek1wantaufik/buddy/model.json) / [Instance](jek1wantaufik/buddy/instance.json) |
| [mousumiroy250/mt5-base/Transformers/default](https://www.kaggle.com/models/mousumiroy250/mt5-base/Transformers/default) | 1 | 6993840981 | [Card](mousumiroy250/mt5-base/model.json) / [Instance](mousumiroy250/mt5-base/instance.json) |
| [muhammadukasha09/bert-base-multilingual-cased/Transformers/default](https://www.kaggle.com/models/muhammadukasha09/bert-base-multilingual-cased/Transformers/default) | 1 | 717272069 | [Card](muhammadukasha09/bert-base-multilingual-cased/model.json) / [Instance](muhammadukasha09/bert-base-multilingual-cased/instance.json) |

Other family-card model sizes or context lengths do not override the required base and 32,768-token limit.
The dependency dataset metadata is stored under `wheelhouse/`; it is separate from the base model.
