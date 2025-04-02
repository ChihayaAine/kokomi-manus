import asyncio
import time
import json
from typing import Optional, Dict, Any, List

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

# Import Talos SDK
from pytalos.client import AsyncTalosClient, SDKScene

# Store notes as a simple key-value dict to demonstrate state management
notes: dict[str, str] = {}

# Store query results
query_results: dict[str, Any] = {}

server = Server("sql_mcp_server")

# Talos client configuration
talos_config = {
    "mis_name": "",
    "session_id": "",
    "doas_group": ""
}

# Initialize Talos client
talos_client = None

def init_talos_client():
    """Initialize Talos client with current configuration"""
    global talos_client
    if talos_config["mis_name"] and talos_config["session_id"]:
        talos_client = AsyncTalosClient(
            talos_config["mis_name"], 
            talos_config["session_id"], 
            sdk_scene=SDKScene.MIS
        )
        talos_client.open_session()
        return True
    return False

@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available note resources and query results.
    Each note is exposed as a resource with a custom note:// URI scheme.
    Each query result is exposed as a resource with a custom query:// URI scheme.
    """
    resources = [
        types.Resource(
            uri=AnyUrl(f"note://internal/{name}"),
            name=f"Note: {name}",
            description=f"A simple note named {name}",
            mimeType="text/plain",
        )
        for name in notes
    ]
    
    # Add query results as resources
    resources.extend([
        types.Resource(
            uri=AnyUrl(f"query://result/{qid}"),
            name=f"Query Result: {qid}",
            description=f"SQL query result with ID {qid}",
            mimeType="application/json",
        )
        for qid in query_results
    ])
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific note's content or query result by its URI.
    """
    if uri.scheme == "note":
        name = uri.path
        if name is not None:
            name = name.lstrip("/")
            return notes[name]
        raise ValueError(f"Note not found: {name}")
    
    elif uri.scheme == "query":
        qid = uri.path
        if qid is not None:
            qid = qid.lstrip("/")
            if qid in query_results:
                return json.dumps(query_results[qid], indent=2)
        raise ValueError(f"Query result not found: {qid}")
    
    raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Each prompt can have optional arguments to customize its behavior.
    """
    return [
        types.Prompt(
            name="summarize-notes",
            description="Creates a summary of all notes",
            arguments=[
                types.PromptArgument(
                    name="style",
                    description="Style of the summary (brief/detailed)",
                    required=False,
                )
            ],
        ),
        types.Prompt(
            name="analyze-query-results",
            description="Analyzes SQL query results",
            arguments=[
                types.PromptArgument(
                    name="query_id",
                    description="ID of the query to analyze",
                    required=True,
                ),
                types.PromptArgument(
                    name="focus",
                    description="Focus of analysis (trends/outliers/summary)",
                    required=False,
                )
            ],
        )
    ]

@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by combining arguments with server state.
    """
    if name == "summarize-notes":
        style = (arguments or {}).get("style", "brief")
        detail_prompt = " Give extensive details." if style == "detailed" else ""

        return types.GetPromptResult(
            description="Summarize the current notes",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Here are the current notes to summarize:{detail_prompt}\n\n"
                        + "\n".join(
                            f"- {name}: {content}"
                            for name, content in notes.items()
                        ),
                    ),
                )
            ],
        )
    
    elif name == "analyze-query-results":
        query_id = (arguments or {}).get("query_id")
        focus = (arguments or {}).get("focus", "summary")
        
        if not query_id or query_id not in query_results:
            raise ValueError(f"Invalid or missing query_id: {query_id}")
        
        focus_prompt = ""
        if focus == "trends":
            focus_prompt = " Focus on identifying trends in the data."
        elif focus == "outliers":
            focus_prompt = " Focus on identifying outliers or anomalies in the data."
        
        return types.GetPromptResult(
            description=f"Analyze query results for {query_id}",
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=f"Please analyze the following SQL query results:{focus_prompt}\n\n"
                        + json.dumps(query_results[query_id], indent=2)
                    ),
                )
            ],
        )
    
    raise ValueError(f"Unknown prompt: {name}")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available tools.
    Each tool specifies its arguments using JSON Schema validation.
    """
    return [
        types.Tool(
            name="configure-talos",
            description="Configure Talos client settings",
            inputSchema={
                "type": "object",
                "properties": {
                    "mis_name": {"type": "string"},
                    "session_id": {"type": "string"},
                    "doas_group": {"type": "string"},
                },
                "required": ["mis_name", "session_id"],
            },
        ),
        types.Tool(
            name="run-sql-query",
            description="Run SQL query using Talos",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "use_doas_group": {"type": "boolean"},
                },
                "required": ["sql"],
            },
        ),
        types.Tool(
            name="get-query-status",
            description="Get status of a running query",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_id": {"type": "string"},
                },
                "required": ["query_id"],
            },
        ),
        types.Tool(
            name="display-query-results",
            description="Display query results in a formatted table",
            inputSchema={
                "type": "object",
                "properties": {
                    "query_id": {"type": "string"},
                    "max_rows": {"type": "integer", "default": 20},
                    "format": {"type": "string", "enum": ["table", "json", "csv"], "default": "table"},
                },
                "required": ["query_id"],
            },
        ),
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle tool execution requests.
    Tools can modify server state and notify clients of changes.
    """
    if name == "configure-talos":
        if not arguments:
            raise ValueError("Missing arguments")
        
        mis_name = arguments.get("mis_name")
        session_id = arguments.get("session_id")
        doas_group = arguments.get("doas_group", "")
        
        if not mis_name or not session_id:
            raise ValueError("Missing mis_name or session_id")
        
        # Update Talos configuration
        talos_config["mis_name"] = mis_name
        talos_config["session_id"] = session_id
        talos_config["doas_group"] = doas_group
        
        # Initialize Talos client
        success = init_talos_client()
        
        return [
            types.TextContent(
                type="text",
                text=f"Talos client {'successfully configured' if success else 'configuration failed'} for user {mis_name}",
            )
        ]
    
    elif name == "run-sql-query":
        if not arguments:
            raise ValueError("Missing arguments")
        
        sql = arguments.get("sql")
        use_doas_group = arguments.get("use_doas_group", False)
        
        if not sql:
            raise ValueError("Missing SQL query")
        
        # Check if Talos client is initialized
        if not talos_client:
            if not init_talos_client():
                return [
                    types.TextContent(
                        type="text",
                        text="Talos client not configured. Please use configure-talos tool first.",
                    )
                ]
        
        try:
            # Submit query
            properties = {}
            if use_doas_group and talos_config["doas_group"]:
                properties["doasGroup"] = talos_config["doas_group"]
            
            qid = talos_client.submit(statement=sql, properties=properties if properties else None)
            
            # Store query ID for status tracking
            query_results[qid] = {"status": "RUNNING", "sql": sql, "submitted_at": time.time()}
            
            # Notify clients that resources have changed
            await server.request_context.session.send_resource_list_changed()
            
            return [
                types.TextContent(
                    type="text",
                    text=f"SQL query submitted successfully. Query ID: {qid}\n\nUse the get-query-status tool to check the status.",
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error submitting SQL query: {str(e)}",
                )
            ]
    
    elif name == "get-query-status":
        if not arguments:
            raise ValueError("Missing arguments")
        
        query_id = arguments.get("query_id")
        
        if not query_id:
            raise ValueError("Missing query_id")
        
        if query_id not in query_results:
            return [
                types.TextContent(
                    type="text",
                    text=f"Query ID {query_id} not found",
                )
            ]
        
        # Check if Talos client is initialized
        if not talos_client:
            if not init_talos_client():
                return [
                    types.TextContent(
                        type="text",
                        text="Talos client not configured. Please use configure-talos tool first.",
                    )
                ]
        
        try:
            # Get query status
            query_info = talos_client.get_query_info(query_id)
            engine_log = talos_client.engine_log(query_id)
            
            # Update query status
            query_results[query_id]["status"] = query_info["status"]
            query_results[query_id]["last_checked"] = time.time()
            
            # If query is finished, fetch results
            if query_info["status"] == "FINISHED":
                res = talos_client.fetch_all(query_id)
                query_results[query_id]["data"] = res["data"]
                query_results[query_id]["columns"] = res.get("columns", [])
                query_results[query_id]["completed_at"] = time.time()
                
                # Notify clients that resources have changed
                await server.request_context.session.send_resource_list_changed()
                
                return [
                    types.TextContent(
                        type="text",
                        text=f"Query completed successfully. Results are now available as a resource.\n\nStatus: {query_info['status']}\nEngine Log: {engine_log}\n\nRows returned: {len(res['data'])}",
                    )
                ]
            elif query_info["status"] in ["QUERY_TIMEOUT", "FAILED", "KILLED"] or query_info["status"].startswith("ERROR_"):
                query_results[query_id]["error"] = engine_log
                query_results[query_id]["completed_at"] = time.time()
                
                # Notify clients that resources have changed
                await server.request_context.session.send_resource_list_changed()
                
                return [
                    types.TextContent(
                        type="text",
                        text=f"Query failed.\n\nStatus: {query_info['status']}\nEngine Log: {engine_log}",
                    )
                ]
            else:
                return [
                    types.TextContent(
                        type="text",
                        text=f"Query is still running.\n\nStatus: {query_info['status']}\nEngine Log: {engine_log}",
                    )
                ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error checking query status: {str(e)}",
                )
            ]
    
    elif name == "display-query-results":
        if not arguments:
            raise ValueError("Missing arguments")
        
        query_id = arguments.get("query_id")
        max_rows = int(arguments.get("max_rows", 20))
        format_type = arguments.get("format", "table")
        
        if not query_id:
            raise ValueError("Missing query_id")
        
        if query_id not in query_results:
            return [
                types.TextContent(
                    type="text",
                    text=f"Query ID {query_id} not found",
                )
            ]
        
        query_result = query_results[query_id]
        
        # 检查查询是否已完成
        if query_result.get("status") != "FINISHED":
            return [
                types.TextContent(
                    type="text",
                    text=f"Query is not finished yet. Current status: {query_result.get('status')}",
                )
            ]
        
        # 检查是否有数据
        if "data" not in query_result or not query_result["data"]:
            return [
                types.TextContent(
                    type="text",
                    text="Query completed but returned no data.",
                )
            ]
        
        data = query_result["data"]
        columns = query_result.get("columns", [])
        
        # 如果没有列名，尝试从第一行数据生成列名
        if not columns and data:
            columns = [f"Column {i+1}" for i in range(len(data[0]))]
        
        # 限制显示的行数
        displayed_data = data[:max_rows]
        
        # 根据请求的格式生成输出
        if format_type == "json":
            # JSON 格式
            formatted_data = []
            for row in displayed_data:
                row_dict = {columns[i]: value for i, value in enumerate(row) if i < len(columns)}
                formatted_data.append(row_dict)
            
            result_text = f"Query results for {query_id} (showing {len(displayed_data)} of {len(data)} rows):\n\n"
            result_text += json.dumps(formatted_data, indent=2)
            
        elif format_type == "csv":
            # CSV 格式
            result_text = f"Query results for {query_id} (showing {len(displayed_data)} of {len(data)} rows):\n\n"
            result_text += ",".join([f'"{col}"' for col in columns]) + "\n"
            for row in displayed_data:
                result_text += ",".join([f'"{str(val)}"' for val in row]) + "\n"
                
        else:
            # 表格格式（默认）
            # 计算每列的最大宽度
            col_widths = [len(str(col)) for col in columns]
            for row in displayed_data:
                for i, val in enumerate(row):
                    if i < len(col_widths):
                        col_widths[i] = max(col_widths[i], len(str(val)))
            
            # 生成表头
            header = "| " + " | ".join(str(col).ljust(col_widths[i]) for i, col in enumerate(columns)) + " |"
            separator = "|-" + "-|-".join("-" * width for width in col_widths) + "-|"
            
            # 生成表格内容
            rows = []
            for row in displayed_data:
                rows.append("| " + " | ".join(str(val).ljust(col_widths[i]) if i < len(col_widths) else str(val) 
                                             for i, val in enumerate(row)) + " |")
            
            result_text = f"Query results for {query_id} (showing {len(displayed_data)} of {len(data)} rows):\n\n"
            result_text += header + "\n" + separator + "\n" + "\n".join(rows)
            
            # 添加总行数信息
            if len(displayed_data) < len(data):
                result_text += f"\n\n*Note: Showing {len(displayed_data)} of {len(data)} total rows. Use 'max_rows' parameter to adjust.*"
        
        return [
            types.TextContent(
                type="text",
                text=result_text,
            )
        ]
    
    raise ValueError(f"Unknown tool: {name}")

async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sql_mcp_server",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )