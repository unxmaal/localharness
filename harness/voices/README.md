# Reference clips for voice cloning

Two clips, shipped with the package so `lh say --voice fr-male` works without a
mounted volume. 626KB and 672KB, which is the price of the feature existing.

## Provenance and licence

Both are from **google/fleurs**, `fr_fr` dev split, licensed **CC-BY-4.0**.
That split was chosen over Common Voice (gated) and Multilingual LibriSpeech
(60GB for French) because its TSV carries the transcript AND a gender label per
utterance, so finding a male French speaker is a filter rather than a listen.

| file | fleurs clip | seconds | transcript |
|---|---|---|---|
| `fr-male.wav` | fleurs-fr-male-3 | 9.8 | Auparavant, le PDG de Ring, Jamie Siminoff, a fait remarquer qu'il a lancé l'entreprise lorsque la sonnette de sa porte d'entrée n'était pas audible depuis son magasin dans son garage. |
| `fr-male-2.wav` | fleurs-fr-male-2 | 10.5 | Les critères qui déterminent une sous-culture comme distincte peuvent être linguistiques, esthétiques, religieux, politiques, sexuels, géographiques ou une combinaison de facteurs. |

227 more male French clips are available from the same split; the full 289-clip
dev split is staged at `/Volumes/Models/corpora/fleurs-fr/` on this machine.

## Why these two

They are the two Eric picked by ear out of three, reading English. The third
(`fleurs-fr-male-1`) is not shipped.

The clips speak FRENCH and are used to say ENGLISH: Chatterbox clones across
languages, and the accent comes with the voice. That is the whole trick, and it
is why no accented-English corpus was needed. Nothing in the Speech Accent
Archive or L2-ARCTIC had to be downloaded.

Note the reference clip is a first-class variable, not a detail: on the French
tts lane these three scored 0.122, 0.144 and 0.578 corpus word error rate.
Audition a new one before adopting it.
