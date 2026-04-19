import json
import time
import sys

from langchain_core.messages import SystemMessage, HumanMessage

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message
from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task


def step_3_llm_item_name_and_rewrite_query(original_query, history_chats):
    """
    根据历史记录识别item_names和重写问题
    :param original_query: 原问题
    :param history_chats:  历史记录
    :return: { item_name = [],rewritten_query = []}
    """
    history_text = ""
    for chat in history_chats:
        history_text += (f"聊天角色：{chat['role']}，回答内容： {chat['text']}，重写问题： {chat['rewritten_query']}，"
                         f"关联主体： {','.join(chat.get('item_names', []))},时间： {chat['ts']}\n")
    prompt = load_prompt('rewritten_query_and_item_names', history_text=history_text, query=original_query)

    llm_model = get_llm_client(json_mode=True)
    messages = [
        HumanMessage(content=prompt)
    ]
    result = llm_model.invoke(messages)
    content = result.content
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    dict_content = json.loads(content)
    if "item_names" not in dict_content:
        dict_content["item_names"] = []
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query
    logger.info(f'完成问题的重写和item_name的提取,结果为:{dict_content}')
    return dict_content


def step_4_query_milvus_item_names(item_names):
    """
    查询向量数据库进行item_name的确定
    :param item_names:  模型提取的item_names可能不准
    :return: [{extracted:模型item_name,matches:[{item_name:xx,score:0.9}]}]
    """
    final_result = []
    milvus_client = get_milvus_client()
    # item_names生成对应的向量(大概率也不会超过max_token,所以放心使用)
    embeddings = generate_embeddings(item_names)
    for index, item_name in enumerate(item_names):
        # 获取当前item_name的向量
        dense_vector = embeddings['dense'][index]
        sparse_vector = embeddings['sparse'][index]
        # 拼接对应的AnnSearchReqeust
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.7, 0.3),
            norm_score=True
        )
        """ 检索结果:
          [
            [
              {id:xx , distance: 0.x,entity:{item_name:xxx} } ,
              {id:xx , distance: 0.x,entity:{item_name:xxx} } 
             ]
          ]
        """
        matches = []  # 当前item对应的匹配结果
        print(response)
        if response and len(response) > 0:
            for hit in response[0]:
                entity = hit.get('entity', {})
                item_name = entity.get('item_name')
                score = hit.get('distance', 0)
                if item_name:
                    matches.append({
                        "item_name": item_name,
                        "score": score
                    })
        final_result.append({
            "extracted": item_names[index],
            "matches": matches
        })
    return final_result


def step_5_confirmed_and_optional_item_name(query_milvus_results):
    """
    通过向量数据库查询的item_name,根据分数归纳出确定和可选的item_name列表
    确定分>0.85>可选>0.60>不要
    :param query_milvus_results:元数据 [{extracted:item_name,matches:[{item_name: , score:},{}]}
                                       ,{extracted:item_name,matches:[{item_name: , score:},{}]}]
    :return: {
             confirmed_item_names:[确定item_name], 分高
             options_item_names:[可选item_name]  分低
          }
    """
    confirmed_item_names = []  # 确定item_name
    options_item_names = []  # 可选item_name
    for item_name_meta in query_milvus_results:
        extracted_name = item_name_meta.get("extracted")
        matches = item_name_meta.get("matches", [])
        # 分数排序
        matches.sort(key=lambda x: x.get('score', 0), reverse=True)
        high_score_matches = [x for x in matches if x.get('score', 0) >= 0.85]
        middle_score_matches = [x for x in matches if x.get('score', 0) >= 0.6]
        # 如果只有一个高分的,直接就返回了,都不需要后续的处理了(肯定是最高分的最好啊)
        if len(high_score_matches) == 1:
            confirmed_item_names.append(high_score_matches[0].get("item_name"))
            continue
        # 如果有多个高分的,优先取同名的,没有同名的才取最高分
        if len(high_score_matches) > 1:
            same_name_item = None
            for item in high_score_matches:
                if item.get("item_name") == extracted_name:
                    same_name_item = item
                    break
            if not same_name_item:
                same_name_item = high_score_matches[0]
            confirmed_item_names.append(same_name_item.get("item_name"))
            continue
        # 没有高分的就走可选逻辑
        if len(middle_score_matches) > 0:
            for item in middle_score_matches:
                options_item_names.append(item.get("item_name"))
            continue
        logger.info(f"没有匹配的item_name，忽略：{extracted_name}")
    # 处理返回结果即可(去重复)
    result = {
        "confirmed_item_names": list(set(confirmed_item_names)),
        "options_item_names": list(set(options_item_names))
    }
    logger.info(f"处理结果为：{result}")
    return result


def step_6_deal_list(state, item_results, history_chats, rewritten_query):
    """
    根据向量查询的结果,判断是否需要赋值answer
    :param state:
    :param item_results:
    :param history_chats:
    :param rewritten_query:
    :return:
    """
    confirmed_item_names = item_results.get("confirmed_item_names", [])
    options_item_names = item_results.get("options_item_names", [])
    if len(confirmed_item_names) > 0:
        # 更新下聊天记录 -》 item_names - > confirmed_item_names (空着)
        # 修改和存储state状态
        state['item_names'] = confirmed_item_names
        state['rewritten_query'] =rewritten_query
        state['history'] = history_chats
        if "answer" in state:
            del state['answer']
        logger.info(f"有确定的item_name:{confirmed_item_names}")
        return state
    # 如果是可选的话就给用户一个选择
    if len(options_item_names) > 0:
        option_names = '、'.join(options_item_names)
        answer = f"您是想咨询以下哪个商品：{option_names}?请下次提问明确商品名称！！"
        state['answer'] = answer
        logger.info(f"有可选的item_name:{options_item_names}")
        return state
    answer = "没有匹配的商品名，请重新提问！！"
    state['answer'] = answer
    logger.info(f"没有匹配的的item_name")
    return state


def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    1.提取item_name(大模型历史对话+本次提问来提取) -> 提取到的item_name向量数据库搜索并打分,根据打分来决定是否采用或进一步处理
    2.利用模型重写问题,提高召回率
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    # 获取历史条件记录
    history_chats = get_recent_messages(session_id=state["session_id"], limit=10)

    # 利用模型提取item_name和重写提问内容
    item_names_and_rewritten_query = step_3_llm_item_name_and_rewrite_query(state["original_query"], history_chats)

    # 向量查询
    item_names = item_names_and_rewritten_query.get("item_names", [])
    rewritten_query = item_names_and_rewritten_query.get("rewritten_query", "")
    item_results = {}
    if len(item_names) > 0:
        query_milvus_results = step_4_query_milvus_item_names(item_names)
        logger.info(f"query_milvus_results:{query_milvus_results}")
        # 查询结果进行处理 区分确定的item_name,可选的item_name和不要的item_name
        item_results = step_5_confirmed_and_optional_item_name(query_milvus_results)

    # 根据向量查询的结果,判断是否需要赋值answer
    state = step_6_deal_list(state, item_results, history_chats, rewritten_query)

    # 记录本次的聊天回答(只记录用户的问题,最终的answer可以再最终的节点保存)
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
    )


    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print(f"---node_item_name_confirm---处理结束")
    return state


if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_006",
        "original_query": "迅饶网关好用吗?",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        logger.exception("==========")
