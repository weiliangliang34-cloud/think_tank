import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.utils.task_utils import add_running_task, add_done_task

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

"""
对于主体识别这个任务来说,文档的前面一些chunk就应该可以确定主体,越到后面可能对于判断"这篇文档是干什么的"的语义偏差越大,所以不要将
全部的chunk直接给模型让它去提炼,而是只给前n个chunk块,
"""

def step_1_get_chunks(state):
    """
    获取chunk和title
    :param state:
    :return:
    """
    chunks = state['chunks']
    file_title = state['file_title']
    if not chunks:
        raise ValueError("chunks没有值，无法继续进行，抛出异常处理！")
    if not file_title:
        # file_title没有值！
        # md_path中获取文件名即可
        file_title = os.path.basename(state.get('md_path'))
        logger.info(f"file_title缺失，获取md_path进行截取！{file_title}")
        state['file_title'] = file_title

    return file_title, chunks


def step_2_build_context(chunks):
    """
    根据chunks切面的content内容进行分拼接！ （2000）
    截取内容限制： 1. 最多截取前top个 （5） 2. 最多字符不能超过 CONTEXT_TOTAL_MAX_CHARS
    截取内容处理：
          切片：{1}，标题:{title},内容：{content} \n\n
          切片：{2}，标题:{title},内容：{content} \n\n
          切片：{3}，标题:{title},内容：{content} \n\n
          切片：{4}，标题:{title},内容：{content} \n\n
          切片：{5}，标题:{title},内容：{content} \n\n
    :param chunks:
    :return:
    """
    parts = []  # 存储处理后的切片['{1},标题:{title},内容:{content}']
    total_chars = 0
    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K], start=1):
        chunks_title = chunk['title']
        chunks_content = chunk['content']
        data = f'切片:{index},标题:{chunks_title},内容:{chunks_content}'
        parts.append(data)
        total_chars += len(data)
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已经达到最大字符数:{total_chars}，停止拼接！")
            break
    context = "\n\n".join(parts)
    final_context = context[:SINGLE_CHUNK_CONTENT_MAX_LEN]
    return final_context


def step_3_recognition_chunk(context, file_title):
    """
    调用模型,获取item_name(主体识别)
    如果获取不到的话使用file_title进行兜底
    :param context:
    :param file_title:
    :return:
    """
    human_prompt = load_prompt('item_name_recognition', file_title=file_title, context=context)
    system_prompt = load_prompt('product_recognition_system')

    llm = get_llm_client(json_mode=False)

    messages = [HumanMessage(human_prompt), SystemMessage(system_prompt)]
    response =  llm.invoke(messages)
    item_name = response.content
    if not item_name:
        item_name = file_title
    return item_name

def step_4_generate_embeddings(item_name):
    vectors = generate_embeddings([item_name])
    dense_vector,sparse_vector = vectors['dense'][0],vectors['sparse'][0]
    return dense_vector,sparse_vector



def step_5_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
    将向量保存到向量数据库中
    :param file_title:
    :param item_name:
    :param dense_vector:
    :param sparse_vector:
    :return:
    """
    # 获取客户端
    milvus_client = get_milvus_client()
    # 判断是否存在集合
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 创建集合
        # 1.创建集合对应的schema(关系型数据库中列的概念)
        schema = milvus_client.create_schema(
            auto_id=True , # 主键自增长
            enable_dynamic_field=True,# 动态字段
        )
        # 2.列信息
        schema.add_field(field_name='pk',datatype=DataType.INT64,is_primary=True,auto_id=True)
        schema.add_field(field_name='file_title',datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name='item_name',datatype=DataType.VARCHAR,max_length=65535)
        schema.add_field(field_name='dense_vector',datatype=DataType.FLOAT_VECTOR,dim=milvus_config.embedding_dim)
        # 稀疏向量不需要配置dim
        schema.add_field(field_name='sparse_vector',datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 3.索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name='dense_vector',
            index_name='dense_vector_index',
            index_type='HNSW', # 索引的算法 ,IVF是聚类(数据多内存少,精度略低,速度慢),HNSW是图层(数据多内存多(2-3倍IVF),精度高,速度快)
            metric_type='COSINE', #向量相似度的计算方式
            params={
                'M':16, # 每个节点最多的邻居节点个数
                'efConstruction':200 #
            }
        )
        """
           10000  M = 16  efConstruction = 200
           50000  M = 32  efConstruction = 300
           100000  M = 64  efConstruction = 400
           M:图中每个节点在层次结构的每个层级所能拥有的最大边数或连接数。M 越高，图的密度就越大，搜索结果的召回率和准确率也就越高，因为有更多的路径可以探索，但同时也会消耗更多内存，并由于连接数的增加而减慢插入时间。
           efConstruction:索引构建过程中考虑的候选节点数量。efConstruction 越高，图的质量越好，但需要更多时间来构建。
        """
        index_params.add_index(
            field_name='sparse_vector',
            index_name='sparse_vector_index',
            index_type='SPARSE_INVERTED_INDEX',
            metric_type='IP',  # 向量相似度的计算方式
            params={'inverted_index_algo':'DAAT_MAXSCORE'}
        )
        # 创建集合
        milvus_client.create_collection(
            collectin_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
    # 要先load一下刷新缓存
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    # 删除原本的数据
    milvus_client.delete(collection_name=milvus_config.item_name_collection,filter=f"item_name=='{item_name}'")

    # 插入数据
    item = {
        'file_title':file_title,
        'item_name':item_name,
        'dense_vector':dense_vector,
        'sparse_vector':sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection,data=[item])
    logger.info(f"保存了item_name:{item_name}的数据到向量数据库中！！")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)
    try:
        # 1. 校验取值(file_title,chunks)
        file_title, chunks = step_1_get_chunks(state)
        # 2.构建上下文环境
        context = step_2_build_context(chunks)

        # 3.识别chunk对应的item_name(主体识别)
        item_name = step_3_recognition_chunk(context, file_title)

        # 修改state中的chunks
        state['item_name'] = item_name
        for chunk in chunks:
            chunk['item_name'] = item_name
        state['chunks'] = chunks

        # 4.生成向量(稠密和稀疏)
        dense_vector, sparse_vector = step_4_generate_embeddings(item_name)

        # 5.保存向量
        step_5_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector)
    except Exception as e:
        logger.error(f">>> [{function_name}]主体识别发生了异常，异常信息：{e}")
        raise  # 终止工作流
    finally:
        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)
    return state


def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()
