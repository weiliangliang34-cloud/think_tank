import base64
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Tuple, List

from minio.deleteobjects import DeleteObject

from app.clients.minio_utils import get_minio_client
from app.conf.lm_config import lm_config
from app.conf.minio_config import minio_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.rate_limit_utils import apply_api_rate_limit
from app.utils.task_utils import add_done_task

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def step_1_valid_state(state) -> tuple[str, Path, Path]:
    """
    校验并且获取本次操作的数据
    :param state: 获取md_path和md_content
    :return: 1. 校验后的md_content  2.md路径对象  3. 获取图片的文件夹 images
    """
    md_path, md_content = state['md_path'], state['md_content']
    if not state['md_path']:
        logger.info(f'md_path为空')
        raise RuntimeError(f'md_path为空')

    md_path_obj = Path(md_path)
    if not md_path_obj:
        raise FileNotFoundError(f'{md_path}文件找不到')

    if not md_content:
        with md_path_obj.open('r', encoding='utf-8') as f:
            md_content = f.read()
        state['md_content'] = md_content

    images_dir = md_path_obj.parent / 'images'

    return md_content, md_path_obj, images_dir


def find_images_in_md_content(md_content, image_file, context_length: int = 100) -> Tuple[str, str] | None:
    """
    根据图片地址和md文件返回图片所在位置的上下文
    :param context_length: 上下文长度
    :param md_content: md内容
    :param image_file: 图片地址
    :return: 图片的附近上文和下文
    """
    """ md_content示例:
        # 图片上文1
        ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)
        图片下文1
        图片上文2
        ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)
        图片下文2
    """
    # 定义正则表达式  .*  .*?
    pattern = re.compile(r"!\[.*?]\(.*?" + image_file + r".*?\)")
    results = []
    # 如果一个md文件中存在一张图片被多处使用,则获取第一个
    for item in pattern.finditer(md_content):
        start, end = item.span()
        pre_text = md_content[max(0, start - context_length):start]  # 上文
        post_text = md_content[end:min(end + context_length, len(md_content))]  # 下文
        results.append((pre_text, post_text))
    # 截取位置前后的内容
    if results:
        logger.debug(f"图片：{image_file} ,使用了：{len(results)}次，截取第一个上下文：{results[0]}")
        return results[0]
    return None


def step_2_recognize_img(md_content, images_path_obj) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    识别md中使用过的图片，采取做下一步（进行图片总结）
    :param md_content: md的内容
    :param images_path_obj: images图片的文件夹地址
    :return: [(图片名,图片地址,(上文,下文))]
    """
    targets = []
    for image_file in os.listdir(images_path_obj):
        if not is_supported_image(image_file):
            logger.warning(f'{image_file}图片格式不正确,无法处理')
            continue
        content_data = find_images_in_md_content(md_content, image_file)
        if not content_data:
            logger.warning(f'{image_file}图片没有在md中使用,上下文为空')
            continue
        targets.append((image_file, str(images_path_obj / image_file), content_data))
    return targets


def step_3_generate_img_summaries(targets, stem) -> dict:
    """
    进行图片内容的总结和处理 （视觉模型）
    :param targets: [(图片名,图片地址,(上文,下文))]
    :param stem: md文件的名称（提示词中 md文件名就是存储图片images的文件名）
    :return: {图片名1:总结,图片名2:总结}
    """
    vl_model = get_llm_client(model=lm_config.vl_model)
    summaries = {}
    limit_deque = deque()
    for (image_file, image_path, (pre_text, post_text)) in targets:
        # 访问限速(1分钟访问10次)
        apply_api_rate_limit(limit_deque, 10)
        # load_prompt('image_summary',root_folder=stem,image_content=[pre_text,post_text])
        prompt = load_prompt('image_summary', **{'root_folder': stem, 'image_content': [pre_text, post_text]})
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # 直接放图片的网络地址 "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                            # base64图片转后的字符串  jpg -> image/jpeg
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            },
        ]
        response = vl_model.invoke(messages)
        summary = response.content.strip().replace("\n", "")
        summaries[image_file] = summary
        logger.debug(f"图片：{image_file}，总结结果：{summary}")
    logger.info(f"总结图片，获取结果：{summaries}")
    return summaries


def step_4_upload_img_minio(summaries, targets, md_content, stem) -> str:
    """
    将原md中的图片替换为图片描述并上传到minio中
    :param summaries: 图片描述{图片名:描述}
    :param targets: (图片名,原地址(上文,下文))
    :param md_content: 原md内容
    :param stem: 文件名
    :return:替换后的md内容
    """
    minio_client = get_minio_client()
    # 1.先删除现有的
    object_list = minio_client.list_objects(minio_config.bucket_name,
                                            prefix=f'{minio_config.minio_img_dir}/{stem}',
                                            recursive=True)
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    errors = minio_client.remove_objects(minio_config.bucket_name,delete_object_list)
    for errors in errors:
        logger.error(f"删除对象失败：{errors}")

    logger.info(f"已经完成{stem}下的对象清空，本次删除了：{len(delete_object_list)}个对象！！！")
    # 2.上传文件
    images_url = {}
    for image_file, image_path, _ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f'{minio_config.minio_img_dir}/{stem}/{image_file}',  # 传入minio桶后面的命名
                file_path=image_path,
                content_type='image/jpeg'
            )
            images_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.debug(f"完成图片{image_file}上传，访问地址为：{images_url[image_file]}")
        except Exception as e:
            logger.error(f"上传图片失败：{image_file}，失败原因：{e}")

    # 3.替换md中的图片
    # 汇总一下需要替换的内容 {图片名:(summary,url)}
    image_infos = {}
    for image_file, summary in summaries.items():
        if url := images_url.get(image_file):
            image_infos[image_file] = (summary, url)
    logger.info(f"图片处理的汇总结果:{image_infos}")

    if image_infos:
        # md内容替换 : ![xx](图片地址/image_file) -> ![summary](minio的url)
        for image_file, (summary, url) in image_infos.items():
            rep = re.compile(r"!\[.*?]\(.*?" + image_file + r".*?\)")
            md_content = rep.sub(f"![{summary}]({url})", md_content)
        logger.debug(f"已经完成md内容的替换，新的内容为:{md_content[:100]}")
    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    完成新的md的磁盘备份，并且返回老地址！
    新的命名  xxx_new.md
    :param new_md_content: 新的内容
    :param md_path_obj: 老地址
    :return: 新地址
    """
    new_md_path = os.path.splitext(md_path_obj)[0] + "_new.md"
    with open(new_md_path, "w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"已经完成了新内容的写入，新的地址为:{new_md_path}")
    return new_md_path


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")

    # 1.准备数据
    md_content, md_path_obj, images_path_obj = step_1_valid_state(state)

    # 2. 扫描 Markdown 中的图片链接。
    if not images_path_obj.exists():
        logger.info(f'{state['md_path']}文件中不存在图片,不需要识别图片')
        return state

    targets = step_2_recognize_img(md_content, images_path_obj)

    # 3. 调用多模态模型生成图片描述。
    summaries = step_3_generate_img_summaries(targets, md_path_obj.stem)

    # 4. 将图片替换为描述并上传替换后的md到minio 并返回替换后的md内容。
    new_md_content = step_4_upload_img_minio(summaries, targets, md_content, md_path_obj.stem)
    # 5. 新的md内容替换和state的修改
    new_md_path = step_5_replace_md_and_save(new_md_content, md_path_obj)
    state['md_content'] = new_md_content
    state['md_path'] = new_md_path
    logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
    add_done_task(state['task_id'], function_name)
    return state


if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT

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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
