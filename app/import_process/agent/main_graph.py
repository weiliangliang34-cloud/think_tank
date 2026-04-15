from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from app.import_process.agent.nodes import node_item_name_recognition
from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
from app.import_process.agent.nodes.node_document_split import node_document_split
from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus
from app.import_process.agent.nodes.node_md_img import node_md_img
from app.import_process.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.import_process.agent.state import ImportGraphState

load_dotenv()

workflow = StateGraph(ImportGraphState)
workflow.add_node("node_entry",node_entry)
workflow.add_node("node_pdf_to_md",node_pdf_to_md)
workflow.add_node("node_md_img",node_md_img)
workflow.add_node("node_document_split",node_document_split)
workflow.add_node("node_item_name_recognition",node_item_name_recognition)
workflow.add_node("node_bge_embedding",node_bge_embedding)
workflow.add_node("node_import_milvus",node_import_milvus)

workflow.set_entry_point("node_entry")

def route_after_entry(state: ImportGraphState) -> str:
    """
    根据文件类型判定第二个节点的路线！
        文件是pdf -> node_pdf_to_md
        文件是md  -> node_md_img
        既不是文件，又不是md  -> END
    :param state: is_md_read_enabled is_pdf_read_enabled
    :return:  node_pdf_to_md | node_md_img | END
    """
    if state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    elif state["is_md_read_enabled"]:
        return "node_md_img"
    else:
        return END

workflow.add_conditional_edges(
         "node_entry",
         route_after_entry,
    {
        "node_pdf_to_md": "node_pdf_to_md",
        "node_md_img": "node_md_img",
        END: END
    })

workflow.add_edge("node_pdf_to_md","node_md_img")
workflow.add_edge("node_md_img","node_document_split")
workflow.add_edge("node_document_split","node_item_name_recognition")
workflow.add_edge("node_item_name_recognition","node_bge_embedding")
workflow.add_edge("node_bge_embedding","node_import_milvus")
workflow.add_edge("node_import_milvus",END)

kb_import_app = workflow.compile()