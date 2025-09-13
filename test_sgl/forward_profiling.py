import argparse
from typing import List, Optional
import torch
import torch.distributed as dist

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator, BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.managers.io_struct import (
    GenerateReqInput, TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ModelWorkerBatch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.sampling.sampling_params import SamplingParams



class ForwardProfiler():
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
    
    def set_extend_input(self, extend_batch: ScheduleBatch, new_extend_input):
        assert len(extend_batch.reqs) == len(new_extend_input)
        for req, extend_input in zip(extend_batch.reqs, new_extend_input):
            # req.prefix_indices = torch.concat([req.prefix_indices, req.fill_ids[len(req.prefix_indices):]])
            req.prefix_indices = torch.tensor(req.fill_ids, dtype=torch.int32)
            req.origin_input_ids.extend(extend_input.tolist())
            # req.fill_ids.extend(extend_input.tolist())
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

    def prefill_then_extend(self, early_extend_input_ids, req_input_ids):

        # sampling_params = [SamplingParams]

        early_bs = len(early_extend_input_ids)
        
        # prepare the first batch
        extend_batch = self.get_schedule_batch_prefill(early_extend_input_ids)
        extend_batch.prepare_for_extend()
        extend_forward_batch = ForwardBatch.init_new(extend_batch.get_model_worker_batch(), self.model_runner)
        

        # forward the first batch
        ret = self.model_runner.forward(extend_forward_batch)

        # prepare the second batch
        extend_input_ids = req_input_ids[:early_bs]
        # extend_input_ids for the early batch
        self.set_extend_input(extend_batch, extend_input_ids)
        # extend_batch.input_ids = extend_input_ids
        # extend_batch.prepare_for_extend()

        # prepare the prefill batch
        prefill_input_ids = req_input_ids[early_bs:]
        prefill_batch = self.get_schedule_batch_prefill(prefill_input_ids)
        # prefill_batch.prepare_for_extend()


        # extend_batch.merge_batch(prefill_batch)  # merge_batch没有merge prefix_lens
        extend_batch = self.merge_extend_batch(extend_batch, prefill_batch)
        # extend_batch.prefix_lens.extend(prefill_batch.prefix_lens)
        extend_batch.prepare_for_extend()
        extend_forward_batch = ForwardBatch.init_new(extend_batch.get_model_worker_batch(), self.model_runner)
        # extend_batch.prepare_for_extend()
        
        # forward the merged prefill & extend batch
        ret = self.model_runner.forward(extend_forward_batch)


        return ret

        # extend_req_indices = early_batch.req_indices
        # process the real input batch
        # test_batch = 
    
    # def forward_batch(self, batch: ForwardBatch):
    #     return self.model_runner.forward_extend(batch)
    

@torch.no_grad()
def generate():
    pass

if __name__ == "__main__":
    # specify lengths of prefill and extend requests
    gpu_id = 0
    tp_rank = 0
    moe_ep_rank = 0
    pp_rank = 0
    model_path = '/home/liux/big_file/Qwen/Qwen3-8B/'
    mem_frac = '0.6'

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

    prefill_bs = 10
    prefill_lens = [100] * prefill_bs
    # prefill_lens = [10] * 5 + [20] * 5
    extend_bs = 20
    # extend_cache_lens = [100] * extend_bs
    extend_cache_lens = [10] * 10 + [20] * 10
    extend_input_lens = [1] * extend_bs

    try:
        early_extend_input_ids, req_input_ids = forward_profiler.prepare_input_ids(
            prefill_lens, extend_cache_lens, extend_input_lens
        )

        # early_batch = forward_profiler.get_forward_batch(
        #     early_extend_input_ids, req_input_ids
        # )

        ret = forward_profiler.prefill_then_extend(
            early_extend_input_ids, req_input_ids
        )

    # print(f'extend_seq_lens: {early_batch.extend_seq_lens}')
    # print(f'input_ids shape: {early_batch.input_ids.shape}')
    # dist.destroy_process_group()
    # exit(0)

        # ret = forward_profiler.forward_batch(early_batch)
    # sampling: model_runner.sample(logits_output, forward_batch)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # raise e
    finally:
        dist.destroy_process_group()


