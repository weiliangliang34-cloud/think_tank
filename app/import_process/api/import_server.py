import shutil
import uuid
from datetime import datetime
from typing import List, Dict, Any

import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette import status

from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.import_process.agent.state import get_default_state
# 项目内部工具/配置/客户端
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)

# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


@app.get("/import", response_class=FileResponse)
async def get_import_file():
    import_html_path = PROJECT_ROOT / 'app' / 'import_process' / 'page' / 'import.html'
    if not import_html_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(path=import_html_path, media_type='text/html')


@app.post('/upload')
async def upload_file(background_tasks: BackgroundTasks,
                      files: List[UploadFile] = File(...)):
    """
      1. 接收文件存储到output文件夹！  /output/当天的日期/uuid(taskid)/文件名
      2. 异步开启，import_graph图的执行 【1. 整个任务的状态（开始和结束） 2. 每个节点的状态 add_running add_done】
    """
    today_str = datetime.now().strftime("%Y%m%d")
    base_out_path = PROJECT_ROOT / 'output' / today_str
    if not base_out_path.exists():
        base_out_path.mkdir(parents=True, exist_ok=True)

    task_ids = []  # 每个上传文件的任务id
    for file in files:
        task_id = str(uuid.uuid4())
        add_running_task(task_id, 'upload_file')
        task_ids.append(task_id)
        # 文件目录
        dir_path = base_out_path / task_id
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        # 具体文件目录+文件名
        local_file_path = dir_path / file.filename
        with local_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 异步执行
        # 参数1: run_import_graph 执行的方法
        # 参数2: *args 参数列表
        background_tasks.add_task(run_import_graph,task_id,str(local_file_path),str(dir_path))
        logger.info(f"{task_id}:完成文件上传，并开启了对应的异步任务！！")
        add_done_task(task_id, "upload_file")
    # 4. 最终返回结果即可
    return {
        "code": 200,
        "message": f"完成了文件上传，并开启了异步任务！文件数量为: {len(files)}",
        "task_ids": task_ids
    }

def run_import_graph(task_id:str,local_file_path:str,local_dir:str):
    """
    开启图的执行和调用
    :param task_id: 每次的标识
    :param local_file_path: 文件的地址
    :param local_dir: 输出文件夹的地址
    :return:
    """
    try:
        update_task_status(task_id,"processing")
        init_state = get_default_state()
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path
        init_state["task_id"] = task_id
        for event in kb_import_app.stream(init_state):
            for node_name, result in event.items():
                logger.info(f"节点：{node_name}已经完成执行，执行结果为：{result}")
        update_task_status(task_id,"completed")
        logger.info(f"{task_id}:图状态执行完毕！！")
    except Exception as e:
        logger.exception("====图执行失败！发生异常====")
        update_task_status(task_id, "failed")


@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO
    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)