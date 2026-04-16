import json
import os
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000  # 512 - 1500 token
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500  # 最小的长度


def step_1_get_content(state):
    """
    读取要切片的内容
    :param state:
    :return:
    """
    md_content = state['md_content']
    if not md_content:
        logger.error(f'无有效的md内容')
        raise Exception(f'无有效的md内容')
    # 处理md_content中的换行符号
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
    file_title = state.get('file_title', 'default_title')
    return md_content, file_title


def step_2_split_by_title(md_content, file_title):
    """
    根据标题切割
    :param md_content: 文本内容
    :param file_title: 标题
    :return: [{content,title,file_title}]
    """
    lines = md_content.split('\n')
    title_pattern = r'^\s*#{1,6}\s+.+'
    current_title = ""
    current_lines = []  # 当前标题行
    title_count = 0
    is_code_block = False  # 是否为代码块
    sections = []  # 最终存储的列表

    for line in lines:
        strip_line = line.strip()
        # 判断是否代码块
        if strip_line.startswith('```') or strip_line.startswith('~~~'):
            is_code_block = not is_code_block  # 这里要取反,不能直接True,因为代码块有上下两个```
            current_lines.append(line)
            continue
        # 判断是否标题
        is_title = (not is_code_block) and re.match(title_pattern, strip_line)
        # 按照 1标题2内容3内容4内容....6标题... 的方式循环,最终会按照标题切分好
        if is_title:
            # 是标题
            # 先检查再存储(只要不是第一次找到标题,就应该先存下来)
            if current_title:
                sections.append({
                    "title": current_title,
                    "content": '\n'.join(current_lines),
                    "file_title": file_title
                })
            current_title = strip_line  # 标题名称
            current_lines = [current_title]
            title_count += 1
        else:
            # 不是标题直接追加
            current_lines.append(line)
    # 最后一个标题的内容保存
    if current_title:
        sections.append({
            "title": current_title,
            "content": "\n".join(current_lines),
            "file_title": file_title
        })
    # 3. 返回结果 sections
    logger.info(f"已经完成chunks的语义粗切！识别chunk数量：{title_count},切片内容:{sections}")
    return sections, title_count, len(lines)


def split_long_section(section, max_length):
    """
    当前chunk如果超长进行循环切割
    :param section: 当前chunk
    :param max_length: 最大长度
    :return: [{},{}]
    """
    content = section['content']
    if len(content) <= max_length:
        return [section]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,
        separators=['\n\n', '\n', '。', '！', "；", " "]
    )
    sub_sections = []
    for index, chunk in enumerate(splitter.split_text(content), start=1):
        logger.info(f'index:{index},chunk:{chunk}')
        text = chunk.strip()  # 切片的内容
        title = f"{section.get('title')}_{index}"
        parent_title = section.get("title")
        part = index
        file_title = section.get("file_title")
        sub_sections.append({
            "title": title,
            "content": text,
            "file_title": file_title,
            "parent_title": parent_title,
            "part": part
        })

    return sub_sections


def merge_short_sections(final_sections, min_length):
    """
    如果切的太小,还需要做合并
    1.content 小于min_lenght
    2.同一个parent_title才能合并
    :param final_sections:
    :param min_length:
    :return:
    """
    merged_sections = []
    pre_section = None
    for section in final_sections:
        if pre_section is None:
            pre_section = section
            continue
        # 判定上一次是否短块需要合并
        is_current_short = len(pre_section.get('content')) < min_length
        # 是否为同一个parent_title
        is_same_parent_tile = pre_section.get("parent_title") and (section.get('parent_title') == pre_section.get('parent_title'))
        if is_current_short and is_same_parent_tile:
            # 需要合并 这里其实需要判断合并后是否会超过最大长度
            current_content = section.get('content')
            pre_section['content'] += '\n\n' + current_content
            pre_section['part'] = section.get('part')
        else:
            # 不是短块或不同父标题,不合并
            merged_sections.append(pre_section)
            pre_section = section
    if pre_section is not None:
        merged_sections.append(pre_section)
    return merged_sections

def step_3_refine_chunks(sections, max_length: int = DEFAULT_MAX_CONTENT_LENGTH, min_length: int = MIN_CONTENT_LENGTH):
    """
    做md内容的精细切割
    1.超过了DEFAULT_MAX_CONTENT_LENGTH 要做切割,(parent_title|part)
    2.小于MIN_CONTENT_LENGTH ,要合并结果 (同一个parent_tile)
    :param sections: 粗切的块
    :param max_length: 最大长度
    :param min_length: 最小长度
    :return:
    """
    final_sections = []
    # 超过的切碎
    for section in sections:
        sub_section = split_long_section(section, max_length)
        final_sections.extend(sub_section)
    # 小于的合并
    final_sections = merge_short_sections(final_sections, min_length)
    # 补全属性part 和parent_tile (没有走split_long_section中切块逻辑的)
    for section in final_sections:
        section['part'] = section.get('part') or 1
        section['parent_tile'] = section.get('parent_tile') or section.get('title')

    return final_sections


def step_4_backup_chunks(local_dir, sections):
    """
    将切割完的碎片进行存储
    :param local_dir: 本地地址
    :param sections: 需要存储的内容
    :return:
    """
    backup_file_path = os.path.join(local_dir,'chunks.json')
    with open(backup_file_path,'w',encoding='utf-8') as f:
        json.dump(
            sections,
            f,
            ensure_ascii=False,
            indent=4,
        )
    logger.info(f"已经将内容,进行备份到:{backup_file_path}")


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f'===[{function_name}]开始,状态为:{state}')
    add_running_task(state['task_id'], function_name)
    try:
        # 1.参数校验
        md_content, file_title = step_1_get_content(state)
        # 2.粗粒度切割 使用标题切割,为了保证语义
        sections, title_count, lines_count = step_2_split_by_title(md_content, file_title)
        # 特殊场景,一个文档没有标题的情况下,给一个默认的标题
        if title_count == 0:
            # 没有标题
            sections = [{'title': '没有主题', 'content': md_content, 'file_title': file_title}]
        # 3.细粒度切割 过大需要设置重叠,过小则合并
        sections = step_3_refine_chunks(sections)
        # 返回数据并备份
        state['chunks'] = sections
        step_4_backup_chunks(state['local_dir'],sections)
    except Exception as e:
        logger.error(f'===[{function_name}]异常,报错:{str(e)}')
        raise
    finally:
        logger.info(f'===[{function_name}]结束,状态为:{state}')
        add_done_task(state['task_id'], function_name)

    return state


if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")
