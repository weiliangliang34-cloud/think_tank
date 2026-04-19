import sys
from app.utils.task_utils import *

from dotenv import load_dotenv
import sys
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
from app.utils.task_utils import add_running_task

load_dotenv()
# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 3
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5  # 最大间断分值


def step_1_merge_rrf_mcp(state):
    """
    进行rrf + mcp数据整合
    :param state:
    :return:
    """
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])

    chunks_list = []
    for chunk in rrf_chunks:
        entity = chunk.get("entity")
        chunk_id = entity.get("chunk_id")
        content = entity.get("content")
        title = entity.get("title")
        chunks_list.append({
            "chunk_id": chunk_id,
            "text": content,
            "title": title,
            "url": "",
            "source": "local",
        })
    for doc in web_search_docs:
        text = doc.get("snippet")
        url = doc.get("url")
        title = doc.get("title")
        chunks_list.append({
            "chunk_id": "",
            "text": text,
            "title": title,
            "source": "web",
            "url": url
        })

    logger.info(f"多路数据融合，最终结果为:{chunks_list}")
    return chunks_list


def step_2_rerank_doc_list(doc_list, state):
    """
    使用rerank进行精排
    :param doc_list:
    :param state:
    :return:
    """
    # 1. 获取原有的问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    # 2. 获取问题对应的所有答案
    text_list = [doc['text'] for doc in doc_list]
    # 3. 加载rerank模型
    rerank = get_reranker_model()
    # 4.处理数据 设置 问题 + 答案 成对
    questions_pairs = [(rewritten_query, text) for text in text_list]
    # 5.精排 normalize=True 归一化 默认False
    scores = rerank.compute_score(questions_pairs, normalize=True)
    doc_list_with_score = []

    for score, item in zip(scores, doc_list):
        item['score'] = score
        doc_list_with_score.append(item)
    # 排序
    doc_list_with_score.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"已经完成排序和打分！最终结果为：{doc_list_with_score}")
    return doc_list_with_score


def step_3_top_k_and_gap(rerank_score_list):
    """
    对rerank模型打分以后得有序集合进行再次算法筛选！
    取出动态的top_k元素即可

    简单的流程总结:

    1.1问题稠密和稀疏向量(问题进行向量混合搜索）
                                                             =》 2.1rrf(同源的排序 rank + weight ) => rrf
    1.2问题+假设性答案稠密和稀疏向量(问题+假设性答案进行向量混合搜索）                                            => 3.rerank  =》 top_k
                                                                2.2mcp
    :param rerank_score_list:
        [
         {
               text:内容 snippet content,
               chunk_id: chunk_id rrf有  mcp None,
               title: title ,
               url : rrfNone mcp url ,
               source: web -> mcp  || local -> rrf ,
               score: rerank打的分
         }
       ]
    :return:
    """
    rerank_max_top_k = RERANK_MAX_TOPK  # 至多获取的元素的数量
    rerank_min_top_k = RERANK_MIN_TOPK  # 至少获取的元素数量
    rerank_gap_abs = RERANK_GAP_ABS  # 断崖的分差    0.9  0.64 =》 0.26 （分）
    rerank_gap_ratio = RERANK_GAP_RATIO  # 断崖的百分比  （1-2）/ 1  =》 0.25 保留
    # 最大比较的元素数量,top_k不应该大于列表长度
    top_k = min(rerank_max_top_k, len(rerank_score_list))
    # 防断崖双指针判断
    if top_k > rerank_min_top_k:
        # 正常循环 取(min-1,top_k-1)
        for index in range(rerank_min_top_k - 1, top_k - 1):
            score_1 = rerank_score_list[index].get("score", 0.0)
            score_2 = rerank_score_list[index + 1].get("score", 0.0)
            # 计算分数的差值gap
            gap = score_1 - score_2
            # 这里的abs是考虑之前没有做归一化导致的负数分数
            rel = gap / (abs(score_1) + 1e-6)
            if gap >= rerank_gap_abs or rel >= rerank_gap_ratio:
                # 断崖分数出现
                logger.info(f"数据集合{index}和{index + 1}的位置发生了断崖，结束循环！！")
                top_k = index + 1  # index下标从0开始 top_k对应的截取长度从1
                break
    else:
        # 如果top_k比rerank_min_top_k小或者等于,正常来说是不会出现的,业务逻辑上就应该限制
        # 即便真的出现了,下面的[:top_k]直接获取(那就不考虑断崖的情况了)
        logger.warning('业务逻辑有问题,需要取的top_k比模型设定的rerank_min_top_k小')

    top_k_doc_list = rerank_score_list[:top_k]
    logger.info(f"最终截取的长度：{top_k},截取的内容:{top_k_doc_list}")
    return top_k_doc_list

def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    logger.info("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1. 非同源路的结果合并 （rrf + mcp） 捏到一个集合中
    """
      [
         rrf = {id:chunk_id,distance:0.x,entity:{chunk_id,content,title}}
         mcp = {snippet: 内容,title:标题,url:关联的文章或者图片的地址}
         {
            text:内容 snippet content,
            chunk_id: chunk_id rrf有  mcp None,
            title: title ,
            url : rrfNone mcp url ,
            source: web -> mcp  || local -> rrf 
         }
      ]

    """
    doc_list = step_1_merge_rrf_mcp(state)

    # 2. 启用rerank进行精排 （数据和分）
    """
    [
      {
            text:内容 snippet content,
            chunk_id: chunk_id rrf有  mcp None,
            title: title ,
            url : rrfNone mcp url ,
            source: web -> mcp  || local -> rrf ,
            score: rerank打的分 
      }
    ]
    """
    rerank_score_list = step_2_rerank_doc_list(doc_list, state)
    # 3. 启动算法进行放断崖以及top_k处理  0.9  0.89  0.35
    final_doc_list = step_3_top_k_and_gap(rerank_score_list)
    # 4. 结果装到state中即可
    state["reranked_docs"] = final_doc_list
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity": {"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity": {"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity": {"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
