import asyncio
from sglang.srt.managers.tokenizer_manager import 

async def async_fetch_data():
    await asyncio.sleep(1)
    return "Data from async world"

def sync_fetch_data():
    # 创建新事件循环或获取现有循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:  # 可能没有当前 loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # 同步等待异步函数完成
    return loop.run_until_complete(async_fetch_data())

# 使用
data = sync_fetch_data()
print(data)  # 输出: Data from async world