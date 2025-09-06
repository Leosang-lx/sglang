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

- 循环：
  - 接收requests：zmq_socket
    - recv_req **=** self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
      - type是什么？
    - 一直收，收不到就停（？）
  - 处理requests
    - TokenizedGenerateReqInput -> self.handle_generate_request()
    - 判断recv_req中的session_params(.id)（这个是什么？）
  - 决定这一轮的forward batch
  - forward：self.run_batch + process_batch_result
  - 记录为last_batch



Request入口：

- GenerateReqInput