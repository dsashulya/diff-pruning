from typing import Optional, List

PAD_TAG = 'PAD'
START_TAG = 'START'
END_TAG = 'END'
UNK_TAG = 'UNK'
OTHER_TAG = 'O'

UTIL_TAGS = [PAD_TAG, START_TAG, END_TAG, UNK_TAG, OTHER_TAG]
ALT_ENTITIES = {'PROTEIN': 'GENE', 'CL': 'CELL_TYPE', 'CHEBI': 'CHEMICAL', 'GGP': 'GENE', 'SPECIES': 'TAXON',
                'CELLLINE': 'CELL_LINE'}
TAB = '\t'


def get_utils_tags_ids(vocab, util_tags: Optional[List[str]] = None) -> List[int]:
    if util_tags is None:
        util_tags = UTIL_TAGS
    util_ids = list(sorted({id_ for tag, id_ in vocab.items() if tag in util_tags}))
    assert max(util_ids) == len(util_ids) - 1  # check all util tags located at the start of vocab
    return util_ids