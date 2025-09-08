import argparse
from typing import List, Optional
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.managers.io_struct import (
    GenerateReqInput, TokenizedGenerateReqInput,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs, PortArgs



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
        early_req_input_ids = None

    else:  # extend requests exit
        assert extend_cache_lens is not None and len(extend_cache_lens) == len(extend_input_lens)
        extend_cache_input_ids = [torch.randint(1, vocab_size, (1, extend_input_len)) for extend_input_len in extend_input_lens]
        extend_input_ids = [torch.randint(1, vocab_size)]

        # generate extend requests (forward before the prefill requests)
        early_requests_obj = GenerateReqInput(
            input_ids=extend_cache_input_ids,   
        )
        early_requests_obj.normalize_batch_and_arguments()

    
    req_input_ids = prefill_req_input_ids + extend_input_ids
    return 
    # todo: 这里应该要转成batch对应上req_id，要不然cache pool没法对应


def init_model_runner(
        model_config: ModelConfig,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        server_args: ServerArgs,
        port_args: PortArgs,
        # is_draft_worker: bool = False,
        # req_to_token_pool: Optional[ReqToTokenPool] = None,
        # token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
):
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=pp_rank,
        pp_size=server_args.pp_size,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
        # is_draft_worker=is_draft_worker,
        # req_to_token_pool=req_to_token_pool,
        # token_to_kv_pool_allocator=token_to_kv_pool_allocator,
    )
    return model_runner

@torch.no_grad()
def generate():
    pass

if __name__ == "__main__":
    # specify lengths of prefill and extend requests
    gpu_id = 0
    tp_rank = 0
    moe_ep_rank = 0
    pp_rank = 0

    prefill_bs = 10
    prefill_lens = [500] * prefill_bs
    extend_bs = 20
    extend_cache_lens = [500] * extend_bs
    extend_input_lens = [1] * extend_bs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(parser.parse_args())
    port_args = PortArgs.init_new(server_args)

    model_config = ModelConfig.from_server_args(
        server_args,
        model_path=server_args.model_path,
    )

    model_runner = init_model_runner(
        model_config,
        gpu_id,
        tp_rank,
        moe_ep_rank,
        pp_rank,
        server_args,
        port_args
    )

    dist.destroy_process_group()


