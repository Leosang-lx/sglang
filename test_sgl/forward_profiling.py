from sglang.srt.managers.scheduler import *
from sglang.srt.managers.tokenizer_manager import *
from sglang.srt.managers.io_struct import (
    GenerateReqInput
)
from typing import List, Optional
import torch


# tokenizer for profiling is unnecessary
def prepare_input_ids(
        vocab_size: int,
        prefill_req_lens: Optional[List[int]]=None,
        extend_cache_lens: Optional[List[int]]=None,
        extend_input_lens: Optional[List[int]]=None,
        ):
    '''
    Each input corresponds to one request.
    params:
    prefill_req_lens: prefill request lengths
    extend_cache_lens: kv cache lengths of extend requests
    extend_input_lens: extend input lengths of extend requests
    return: input_ids: input ids for prefill and extend requests
    '''
    if (prefill_req_lens is None or len(prefill_req_lens) == 0) and \
        (extend_cache_lens is None or len(extend_cache_lens) == 0) and \
        (extend_input_lens is None or len(extend_input_lens) == 0):
        return None
    # at least one prefill request or extend request
    
    # prepare prefill inputs
    if prefill_req_lens is None or len(prefill_req_lens) == 0:
        prefill_req_input_ids = []
    else:
        prefill_req_input_ids = [torch.randint(1, vocab_size, (1, prefill_req_len)) for prefill_req_len in prefill_req_lens]

    # prepare extend inputs
    if extend_input_lens is None or len(extend_input_lens) == 0:
        extend_input_ids = None

    else:  # extend requests exit
        assert extend_cache_lens is not None and len(extend_cache_lens) == len(extend_input_lens)
        extend_cache_input_ids = [torch.randint(1, vocab_size, (1, extend_input_len)) for extend_input_len in extend_input_lens]
        extend_input_ids = [torch.randint(1, vocab_size)]

        # generate extend requests (forward before the prefill requests)
        early_requests_obj = GenerateReqInput(
            
        )
    
    req_input_ids = prefill_req_input_ids + extend_input_ids

    # todo: 这里应该要转成batch对应上req_id，要不然cache pool没法对应


