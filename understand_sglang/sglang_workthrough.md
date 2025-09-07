bench_offline_serving的request全过程

bench_offline_serving: throughput_test_once line()

- backend.generate(prompt=prompt， sampling_params=sampling_params) line 217

engine: generate()

- 生成obj=GenerateReqInput(input_prompt/ids, sampling_params)
- (async) generator=self.tokenzier_manager.generate_request(obj, None)
- 使用loop.run_until_complete同步执行generator

tokenizer_manager: 

- obj.normalize_batch_and_arguments()
- self.\_send_one_request()或者self.\_handle_batch_request()
- if parallel\_sample_num == 1:
  - self.\_batch_tokenize_and_process(batch\_size, obj)
    - 在这里进行tokenizer(texts)
    - self._create_tokenized_object(obj)
      - GenerateReqInput -> TokenizedGenerateReqInput
  - 逐个发给Scheduler，并将对应等待response的协程存在generators中
    - send TokenizedGenerateReqInput
    - GenerateReqInput用于记录ReqState来_wait_one_response
  - tokenizer_manager到这里应该就结束了，到Scheduler

**scheduler**: 主行为入口event_loop_xxx()：同步

- 循环batch forward：
  - 接收requests：zmq_socket
    - recv_req **=** self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
      - type是什么？
    - 一直收，收不到就停（？应该有个异步队列吧）
  - 处理requests
    - TokenizedGenerateReqInput -> self.handle_generate_request()
      - 判断recv_req中的session_params(.id)（这个是什么？）
      - 基于recv_req中的信息生成req = **Req**()（类型转换）
      - prefetch_kvcache + append **Req** to waiting_queue
  - 决定这一轮的forward batch: get_next_batch_to_run()
    - 如果有prefill，优先处理
    - 把完成prefill的request放进running_batch: running_batch即可以进行decode的batch
      - 判断该轮是否有prefill请求：preill & chunked prefill: get_new_prefill_batch
      - 如果有prefill_batch不为空，则返回prefill_batch: **ScheduleBatch**
      - 否则返回running_batch: **ScheduleBatch**
  - forward：self.**run_batch**(batch: **ScheduleBatch**) + process_batch_result
  - 记录为last_batch

**Forward**: 前向传播batch forward
问题：这里的cache是如何管理和更新的？
Cache: model_runner.req_to_token_pool & token_to_kv_pool_allocator
scheduler.run_batch(ScheduleBatch)
- batch.get_model_worker_batch(): **ScheduleBatch** -> **ModelWorkerBatch**
- self.tp_worker.forward_batch_generation(model_worker_batch) -> hidden_states & logits_output
- last_rank: GenerationBatchResult(hidden_states & logits_output)

tp_worker.forward_batch_generation(ModelWorkerBatch)
- **ModelWorkerBatch** -> **ForwardBatch**,包含了model_runner的attn_backend
- tp_worker.model_runner.forward(forward_batch, pp_proxy_tensors)，这是PP的中间结果
- 如果是last_rank：额外进行采样：next_token_ids = self.model_runner.sample(logits_output)
  - 不是则next_token_ids = None
  
model_runner.forward(ForwardBatch, pp_proxy_tensors)
- cuda_graph相关的：不懂，，，
- 根据forward_batch.forward_mode，选择对应的forward方法（以prefill&extend为例）
- self.forward_extend(ForwardBatch, pp_proxy_tensors)
  - attn_backend.init_forward_metadata(forward_batch)
    - 设置attn_backend所需的元数据并储存,这个会在下面attn_backend.forward()中用到
  - self.model_runner.model.forward(forward_batch.input_ids, positions, forward_batch)

model.forward()
这部分基本跟transformers中的定义差不多,除了input加入了forward_batch和attention部分
哪里调用的attention_backend?哪里涉及到kv_cache?是否支持原生cache?
以Qwen3为例
- Qwen3ForCausalLM.forward(input_ids, positions, forward_batch)
  - hidden_states = Qwen3Model(Qwen2Model).forward(input_ids, positions, forward_batch)
    - embedding & pp_proxy_tensors -> layer(positions, hidden_states, forward_batch, residual)
    - decoder_layer -> Qwen3Attention + Qwen3MLP + RMSNorm
      - Qwen3Attention -> QKVproj + split -> norm -> rope -> self.attn(q,k,v,**forward_batch**)
        - Qwen3Attention.attn = RadixAttention: SGLang中attention的基类
        - RadixAttention.forward()

attn_backend.forward(q, k, v, layer, forward_batch, save_kv_cache)
- __init__()初始化
  - prefill_wrappers_ragged
  - prefill_wrappers_paged
  - prefill_wrappers_verify: speculative decoding
  - decode_wrappers: decode
- ForwardBatch:
  - Requests
    - req_pool_indices: torch.Tensor
  - AttentionBackend相关参数
    - req_to_token_pool: ReqToTokenPool
    - token_to_kv_pool: KVCache
    - attn_backend
- AttentionBackend.forward() -> self.forward_decode() & self.forward_extend()
  - 以prefill & extend为例
  - 之前在model_runner中调用forwar_extend()已经init_forward_metadata()了
  - 从forward_metadata中获取prefill_wrapper
    - prefill & extend时use_ragged=True
  - forward_batch.token_to_kv_pool.set_kv_buffer(): 设置新的kv
  - o = prefill_wrapper_paged.forward(q, token_to_kv_pool.get_kv_buffer(layer.layer_id),...)
  - (再底层的就先不管了,看不懂)
