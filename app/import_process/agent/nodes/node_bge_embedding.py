import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.utils.task_utils import add_running_task, add_done_task


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    current_node = sys._getframe().f_code.co_name
    add_running_task(state.get("task_id", ""), current_node)
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")
    try:
        # 获取要生成向量的chunks
        chunks = state['chunks']
        # 每个chunk生成向量
        # 需要拼接成 f"商品名:{},介绍:{}"
        final_chunks = [] # 存储处理完的chunk(带有向量)
        batch_size = 5
        for i in range(0,len(chunks),batch_size):
            batch_items = chunks[i:i+batch_size]
            current_texts = []
            for item in batch_items:
                item_name = item.get('item_name')
                item_content = item.get('content')
                item_text = f'商品:{item_name},内容介绍:{item_content}'
                current_texts.append(item_text)

            result = generate_embeddings(current_texts)
            for i,chunk in enumerate(batch_items):
                chunk_item = chunk.copy()
                chunk_item['dense_vector'] = result['dense'][i]
                chunk_item['sparse_vector'] = result['sparse'][i]
                final_chunks.append(chunk_item)
        state['chunks'] = final_chunks
        logger.info(f"--- BGE-M3 向量化处理完成，共处理 {len(final_chunks)} 条文本切片 ---")
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)

    return state