"""Document split for the write-feature steering study (2026-09-17): the cue-token means are built from the TRAINING documents of a
family, the 0-shot steering test uses the HELD-OUT documents, so no test prompt ever contributes to its own vector."""
import zlib


def heldout(doc_id):
    """~25 % of the documents (deterministic in the doc id): 0-shot steering test set."""
    return zlib.crc32(doc_id.encode()) % 4 == 0
