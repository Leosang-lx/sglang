import argparse
import time
from typing import List, Optional, Generator
# from types import GeneratorType
import torch
import torch.distributed as dist
from profiler.profiler import prof
from contextlib import nullcontext

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.managers.io_struct import (
    GenerateReqInput, TokenizedGenerateReqInput,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ModelWorkerBatch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.sampling.sampling_params import SamplingParams



class ForwardProfiler():
    '''
    **Single-gpu** profiling for now
    '''
    def __init__(
        self,
        # model_runner: ModelRunner,
        # model_config: ModelConfig,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        server_args: ServerArgs,
        port_args: PortArgs,
        tp_size: int = 1,
        pp_size: int = 1,
    ):
        self.server_args = server_args
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
        )
        self.enable_kv_cache_events = server_args.kv_events_config is not None
        
        self.init_model_runner(
            self.model_config,
            gpu_id,
            tp_rank,
            moe_ep_rank,
            pp_rank,
            server_args,
            port_args
        )
        # cache_allocation
        self.req_to_token_pool = self.model_runner.req_to_token_pool
        self.token_to_kv_pool_allocator = self.model_runner.token_to_kv_pool_allocator

        self.tree_cache = RadixCache(
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            page_size=self.server_args.page_size,
            disable=server_args.disable_radix_cache,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
    
    def init_model_runner(
            self,
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
        self.model_runner = model_runner

    # tokenizer for profiling is unnecessary
    def prepare_input_ids(
        self,
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

        vocab_size = self.model_config.vocab_size
        
        # prepare prefill inputs
        if prefill_req_lens is None or len(prefill_req_lens) == 0:
            prefill_req_input_ids = []
        else:
            prefill_req_input_ids = [torch.randint(1, vocab_size, (prefill_req_len,)) for prefill_req_len in prefill_req_lens]

        # prepare extend inputs
        if extend_input_lens is None or len(extend_input_lens) == 0:
            early_extend_input_ids = None
            extend_req_input_ids = []

        else:  # extend requests exit
            assert extend_cache_lens is not None and len(extend_cache_lens) == len(extend_input_lens)
            early_extend_input_ids = [torch.randint(1, vocab_size-10, (extend_cache_len,)) for extend_cache_len in extend_cache_lens]
            extend_req_input_ids = [torch.randint(1, vocab_size-10, (extend_input_len,)) for extend_input_len in extend_input_lens]

        req_input_ids = extend_req_input_ids + prefill_req_input_ids
        
        # 这里的req是对应的
        # 如果early_extend_input_ids有k个batch
        # 那么req_input_ids中的前k个extend_input_ids就是对应的extend_input_ids
        return early_extend_input_ids, req_input_ids

    def prepare_reqs(self, input_ids: list[torch.Tensor]):
        assert input_ids is not None and len(input_ids) > 0
        reqs = []
        for i in range(len(input_ids)):
            req = Req(
                rid=str(i),
                origin_input_text=None,
                origin_input_ids=input_ids[i].tolist(),
                sampling_params=SamplingParams(temperature=0.0),  # prepare sampling_params: SamplingParams
                # return_logprob=False,
                # top_logprobs_num=self.top_logprobs_num,
                # token_ids_logprob=self.token_ids_logprob,
            )
            req.init_next_round_input()
            reqs.append(req)

        return reqs


    def get_schedule_batch_prefill(self, req_input_ids):
        reqs = self.prepare_reqs(req_input_ids)
        batch = ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            False,
            None,
            self.server_args.enable_custom_logit_processor,
            # chunked_req=self.chunked_req,
        )

        return batch
    
    def process_batch(self, batch: ScheduleBatch, result: LogitsProcessorOutput):
        next_token_ids = result.next_token_ids if hasattr(result, 'next_token_ids') else None
        for i, req in enumerate(batch.reqs):
            if req.is_retracted:
                continue
            if self.tree_cache.disable:
                req.prefix_indices = torch.tensor(req.fill_ids)
            else:
                if req.is_chunked <= 0:
                    if next_token_ids is not None:
                        req.output_ids.append(next_token_ids[i])
                        req.check_finished()

                        if req.finished():
                            self.tree_cache.cache_finished_req(req)
                            req.time_stats.completion_time = time.time()
                        else:
                            self.tree_cache.cache_unfinished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
    
    def set_extend_input(self, extend_batch: ScheduleBatch, new_extend_input):
        assert len(extend_batch.reqs) == len(new_extend_input)
        for req, extend_input in zip(extend_batch.reqs, new_extend_input):
            req.origin_input_ids.extend(extend_input.tolist())
            # req.output_ids.extend(extend_input.tolist())
            req.init_next_round_input()

    def merge_extend_batch(self, batch1: ScheduleBatch, batch2: ScheduleBatch):
        # simplipied ScheduleBatch.merge_batch() for extend forward profiling
        # before prepare_for_extend()

        # batch1.sampling_info.merge_batch(batch2.sampling_info)
        batch1.reqs.extend(batch2.reqs)
        if batch1.spec_info:
            batch1.spec_info.merge_batch(batch2.spec_info)

        return batch1
        # batch1.req_pool_indices = torch.cat(
        #     [batch1.req_pool_indices, batch2]
        # )

    def prefill_then_extend(self, early_extend_input_ids, req_input_ids, prof=None, name=None):

        if early_extend_input_ids is not None:  # extend requests exist
            # sampling_params = [SamplingParams]
            extend_bs = len(early_extend_input_ids)
            
            # prepare the first batch
            extend_batch = self.get_schedule_batch_prefill(early_extend_input_ids)
            extend_batch.prepare_for_extend()
            extend_forward_batch = ForwardBatch.init_new(extend_batch.get_model_worker_batch(), self.model_runner)

            # forward the first batch
            ret, _ = self.model_runner.forward(extend_forward_batch)

            # process the first batch
            self.process_batch(extend_batch, ret)

            # prepare the second batch
            extend_input_ids = req_input_ids[:extend_bs]
            # extend_input_ids for the early batch
            self.set_extend_input(extend_batch, extend_input_ids)
        else:
            extend_bs = 0

        if extend_bs < len(req_input_ids):  # prefill reqs exist
            # prepare the prefill batch
            prefill_input_ids = req_input_ids[extend_bs:]
            prefill_batch = self.get_schedule_batch_prefill(prefill_input_ids)

            if extend_bs == 0:  # only prefill reqs
                extend_batch = prefill_batch
                
            else:  # also have extend reqs
                # merge batch/reqs
                extend_batch = self.merge_extend_batch(extend_batch, prefill_batch)
            
        extend_batch.prepare_for_extend(save_cache=False)
        extend_forward_batch = ForwardBatch.init_new(extend_batch.get_model_worker_batch(), self.model_runner)

        name = name if name else 'batch forward'
        
        # forward the merged prefill & extend batch
        for _ in range(10):
            with prof.profile_context(name, device=f'cuda:{gpu_id}') if prof else nullcontext():
                ret, _ = self.model_runner.forward(extend_forward_batch)

        return ret
    
    def clear_cache(self):
        self.tree_cache.reset()
        self.req_to_token_pool.clear()
        self.token_to_kv_pool_allocator.clear()

def prepare_warmup_input():
    # extend
    bs = 10
    extend_cache_len = 100
    extend_req_input_len = 100
    early_extend_input_ids, req_input_ids = forward_profiler.prepare_input_ids(
        [], [extend_cache_len] * bs, [extend_req_input_len] * bs
    )
    yield (early_extend_input_ids, req_input_ids), 'warmup'


def prepare_single_prefill_forward_input():
    prefill_lens = [1]
    prefill_lens.extend(list(range(100, 2001, 100)))
    for prefill_len in prefill_lens:
        early_extend_input_ids, req_input_ids = forward_profiler.prepare_input_ids(
            [prefill_len], None, None
        )
        prof_name = f'prefill_len={prefill_len} forward'
        yield (early_extend_input_ids, req_input_ids), prof_name

def prepare_batched_decode_forward_input():
    # extend_bs = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    extend_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    extend_cache_len = 127
    extend_req_input_len = 1
    for bs in extend_bs: 
        extend_cache_lens = [extend_cache_len] * bs
        extend_req_input_lens = [extend_req_input_len] * bs
        early_extend_input_ids, req_input_ids = forward_profiler.prepare_input_ids(
            [], extend_cache_lens, extend_req_input_lens
        )
        prof_name = f'batch decode {bs}*({extend_cache_len}+={extend_req_input_len}) forward'
        yield (early_extend_input_ids, req_input_ids), prof_name

def forward_profiling(input_generators: Generator, forward_profiler: ForwardProfiler, prof=None):
    for (early_extend_input_ids, req_input_ids), prof_name in input_generators:
        forward_profiler.clear_cache()
        ret = forward_profiler.prefill_then_extend(
            early_extend_input_ids, req_input_ids, prof, prof_name
        )


if __name__ == "__main__":
    # specify lengths of prefill and extend requests
    gpu_id = 0  # set located gpu
    tp_rank = 0
    moe_ep_rank = 0
    pp_rank = 0
    
    prefix = '/home/liux/big_file'

    # model_id = 'Qwen/Qwen3-8B'
    # model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
    model_id = 'lmsys/vicuna-13b-v1.3'
    # model_id = 'meta-llama/Llama-2-13b-chat-hf'
    # model_id = 'meta-llama/Llama-3.3-70B-Instruct'

    model_path = f'{prefix}/{model_id}'

    mem_frac = '0.5'  # max-gpu-mem-usage

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    server_args = ServerArgs.from_cli_args(
        # parse args in python
        parser.parse_args(args=[
            '--model-path', model_path,
            '--mem-fraction-static', mem_frac,
            '--disable-radix-cache',
            ])
    )
    port_args = PortArgs.init_new(server_args)

    forward_profiler = ForwardProfiler(
        gpu_id,
        tp_rank,
        moe_ep_rank,
        pp_rank,
        server_args,
        port_args,
    )
    try:
        # warmup
        print('Warmup...')
        warm_up_input = prepare_warmup_input()
        forward_profiling(warm_up_input, forward_profiler)

        # profiling
        print('Profiling...')
        # prefill_input_g = prepare_single_prefill_forward_input()
        # forward_profiling(prefill_input_g, forward_profiler, prof)
        extend_input_g = prepare_batched_decode_forward_input()
        forward_profiling(extend_input_g, forward_profiler, prof)


    # sampling: model_runner.sample(logits_output, forward_batch)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # raise e
    finally:
        dist.destroy_process_group()
    
    prof.print_all_events()


