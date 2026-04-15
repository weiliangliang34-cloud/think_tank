import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task


def step_1_validate_paths(state) -> tuple[Path, Path]:
    """
    进行路径校验 pdf_path失效,直接异常处理
                local_dir 没有的话给默认值
    :param state:
    :return:
    """
    logger.debug(f">>> [step_1_validate_paths]在md转pdf下，开始进行文件格式校验！！")
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    # 常规的非空校验 （站在字符串的角度）
    if not pdf_path:
        logger.error("step_1_validate_paths检查发现没有输入文件，无法继续解析！！")
        raise ValueError("step_1_validate_paths检查发现没有输入文件，无法继续解析！！")
    if not local_dir:
        # 给与一个输出的默认值
        local_dir = str(PROJECT_ROOT / "output")
        logger.info(f"step_1_validate_paths检查发现local_dir没有赋值，给与默认值：{local_dir}！")
    # 进行文件存在校验
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths检查发现pdf_path不存在，请检查输入文件路径是否正确！！")
        raise FileNotFoundError(f"[step_1_validate_paths]检查发现pdf_path不存在，请检查输入文件路径是否正确！！")
    if not local_dir_obj.exists():
        logger.error(f"[step_1_validate_paths检查发现local_dir不存在，主动创建对应的文件夹！！！")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


def step_2_upload_and_poll(pdf_path_obj) -> Any | None:
    """
    将pdf文件使用minerU解析,并获取md文件下载的url
    :param pdf_path_obj: 上传解析pdf文件的Path对象
    :return: 下载的url地址
    """
    # 1.先请求MinerU获取一个可PUT请求的OSS地址
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {
                "name": f'{pdf_path_obj.name}',
            }
        ],
        "model_version": "vlm"
    }

    res = requests.post(url, headers=header, json=data)
    logger.info(f"MinerU请求url:{url},返回状态码:{res.status_code},返回结果:{res.json()}")
    if res.status_code != 200 or res.json()['code'] != 0:
        logger.error(f'MinerU请求url:{url}失败,异常:{res.json()}')
        raise RuntimeError(f'MinerU请求url:{url}失败,异常:{res.json()}')
    # 本身MinerU支持批量上传,但是这个方法只传一个,所以取0的就行
    mineru_upload_url = res.json()["data"]["file_urls"][0]
    batch_id = res.json()["data"]["batch_id"]

    # 2.使用PUT请求上传文件到对应的解析地址
    http_session = requests.Session()
    http_session.trust_env = False  # 禁止走代理,复用请求对象
    try:
        with open(pdf_path_obj, 'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(mineru_upload_url, data=file_data)
        if upload_response.status_code != 200:
            logger.error(f'MinerU上传文件{pdf_path_obj.name}到{mineru_upload_url}失败')
            raise RuntimeError(f'MinerU上传文件{pdf_path_obj.name}到{mineru_upload_url}失败')
    except Exception as e:
        logger.error(f'MinerU上传文件{pdf_path_obj.name}到{mineru_upload_url}失败')
        raise RuntimeError(f'MinerU上传文件{pdf_path_obj.name}到{mineru_upload_url}失败')
    finally:
        http_session.close()

    # 3.轮询获取结果
    # 循环获取,确保获取到结果再执行,3秒获取一次,最多等待10分钟
    url = f'{mineru_config.base_url}/extract-results/batch/{batch_id}'
    timeout_seconds = 600  # 超时600秒
    poll_interval = 3  # 间隔3秒
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout_seconds:
            logger.error(f'MinerU轮询结果{url}超时')
            raise TimeoutError(f'Mineru轮询结果{url}超时')
        res = requests.get(url, headers=header)
        logger.debug(f'MinerU返回结果:{res}')
        if res.status_code != 200:
            if 500 <= res.status_code <= 599:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f'MinerU返回结果:{res},非5XX系列异常不再轮询')

        json_data = res.json()
        if json_data['code'] != 0:
            raise RuntimeError(f'MinerU返回结果:{res},解析异常')
        extract_result = json_data['data']['extract_result'][0]
        if extract_result['state'] == 'done':
            zip_url = extract_result['full_zip_url']
            logger.info(f'MinerU解析完成,耗时:{time.time() - start_time}s,zip_url:{zip_url}')
            return zip_url
        else:
            time.sleep(poll_interval)


def step_3_download_and_extract(zip_url, local_dir_obj: Path, stem) -> str:
    """
    下载指定的md.zip文件,并且解压,返回解压后的md文件
    :param zip_url: 要下载的地址
    :param local_dir_obj: 存储的文件夹
    :param stem: pdf的文件名字
    :return: 返回md文件的地址
    """
    # 1.下载zip
    res = requests.get(zip_url)
    if res.status_code != 200:
        logger.error(f'{zip_url}下载失败')
        raise RuntimeError(f'{zip_url}下载失败')

    # 2.保存zip到本地
    zip_save_path = local_dir_obj / f'{stem}_result.zip'
    with open(zip_save_path, 'wb') as f:
        f.write(res.content)
    logger.info(f'{zip_url}下载文件保存的位置{zip_save_path}')

    # 3.清空旧目录并解压新文件
    extract_target_dir = local_dir_obj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    extract_target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_save_path, 'r') as zip_ref:
        zip_ref.extractall(extract_target_dir)

    # 4.返回md文件的地址
    # 解压后的文件可能叫 stem.md 也可能叫 full.md(低版本) 所以做个判断
    md_file_list = list(extract_target_dir.rglob('*.md'))
    if not md_file_list:
        logger.error(f'md_file_list为空')
        raise RuntimeError(f'md_file_list为空')
    target_md_file = None  # 存储最终的md文件
    # 4.1检查有stem.md文件
    for md_file in md_file_list:
        if md_file.name == stem + ".md":
            target_md_file = md_file
            break
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == 'full.md':
                target_md_file = md_file
                break
    if not target_md_file:
        target_md_file = md_file_list[0]
    # md文件名->可能为 {stem}.md 或 full.md 或 不知道什么命名.md
    if target_md_file.stem != stem:
        # 重命名成 {stem}.md
        target_md_file  = target_md_file.rename(target_md_file.with_name(f'{stem}.md'))

    logger.info(f'最终存储md路径:{target_md_file.resolve()}')
    return target_md_file.resolve()


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
        1. 进入的日志和任务状态的配置
        2. 进行参数校验 （local_dir -》 给与默认值 | local_file_path完成字面意思的校验 -》 深入校验校验的文件是否真的存在）
        3. 调用minerU进行pdf的解析（local_file_path）返回一个下载文件的地址 xx.zip url地址
        4. 下载zip包，并且解析和提取 （local_dir）
        5. 把md_path地址进行赋值，读取md的文件内容 md_content赋值（文本内容）
        6. 结束的日志和任务状态的配置
        容错率处理！！ try异常处理
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f'===[{function_name}]开始,状态为:{state}')
    add_running_task(state['task_id'], function_name)
    try:
        # 校验,返回可以直接使用的路径对象
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
        # 使用minerU进行pdf解析,返回一个下载文件的地址 xx.zip的url地址
        zip_url = step_2_upload_and_poll(pdf_path_obj)
        # 下载zip文件,并解析和提取
        md_path = step_3_download_and_extract(zip_url, local_dir_obj, pdf_path_obj.stem)

        # 更新数据
        state['md_path'] = md_path
        state['local_dir'] = str(local_dir_obj)
        with open(md_path, 'r', encoding='utf-8') as f:
            state['md_content'] = f.read()
    except Exception as e:
        logger.error(f'===[{function_name}]异常,报错:{str(e)}')
        raise
    finally:
        logger.info(f'===[{function_name}]结束,状态为:{state}')
        add_done_task(state['task_id'], function_name)

    return state

if __name__ == "__main__":
    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")