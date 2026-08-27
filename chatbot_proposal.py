import os
from typing import Annotated, Sequence, TypedDict, Literal
import chainlit as cl
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import END, StateGraph, START
from pymilvus import MilvusClient
from llama_stack_client import LlamaStackClient

load_dotenv()


def _float_env(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _int_env(name: str, default: str) -> int:
    return int(os.getenv(name, default))


base_url = os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")
api_key = os.getenv("API_KEY", "fake-key")
model_id = os.getenv("MODEL_ID", "granite-3-3-8b-instruct")
embedding_model = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/ibm-granite/granite-embedding-125m-english",
)
coordinator_agent_instructions = os.getenv(
    "COORDINATOR_AGENT_INSTRUCTIONS",
    (
        "Eres el Director de Orquestación de IA. Tu trabajo es derivar las dudas del usuario "
        "al especialista correcto.\n\n"
        "PASO 1 — Identifica los dominios de la consulta:\n"
        "* Dominio SKUs: códigos de producto (MW0xxxx), catálogo, precios, licencias, "
        "ofertas/propuestas comerciales, qué SKU usar o recomendar.\n"
        "* Dominio Infraestructura: tipos de infraestructura (privada, on-prem, bare-metal, "
        "VMware, nube pública/híbrida), compatibilidad, despliegue, arquitectura de cluster, "
        "nodos, topología.\n\n"
        "PASO 2 — Reglas de ruteo según dominios detectados:\n"
        "A) Si la consulta pide AMBOS dominios (infraestructura Y SKUs), debes usar los DOS "
        "agentes en este turno. Orden recomendado: primero 'agente_infraestructura', luego "
        "'agente_skus'. No elijas FIN hasta que ambos hayan participado.\n"
        "B) Si la consulta es SOLO de SKUs sin pedir recomendación de infraestructura, asigna "
        "únicamente 'agente_skus'.\n"
        "C) Si la consulta es SOLO de infraestructura sin pedir SKUs ni precios, asigna "
        "únicamente 'agente_infraestructura'.\n"
        "D) Si la consulta mezcla SKUs con vCPUs/cores/licencias pero NO pide infraestructura "
        "ni plataformas, asigna solo 'agente_skus'.\n\n"
        "PASO 3 — Control del flujo:\n"
        "1. NUNCA reasignes un agente que ya completó su intervención en este turno.\n"
        "2. Si un agente ya respondió pero la consulta requería otro dominio pendiente, "
        "deriva al agente faltante (no elijas FIN).\n"
        "3. Elige FIN solo cuando todos los dominios requeridos ya fueron atendidos."
    ),
)
coordinator_agent_delegator_prompt = os.getenv(
    "COORDINATOR_AGENT_DELEGATOR_PROMPT",
    (
        "¿Quién debe actuar ahora? Revisa si la consulta del usuario requiere infraestructura, "
        "SKUs o ambos. Si faltan agentes por ejecutar en este turno, deriva al pendiente. "
        "Selecciona estrictamente una opción: {options}"
    ),
)
skus_agent_instructions = os.getenv(
    "SKUS_AGENT_INSTRUCTIONS",
    "Eres el Agente Experto en SKUs y Catálogo de Red Hat. Tu única fuente de verdad es la "
    "herramienta Milvus. Úsala siempre para validar códigos de producto, inventario y precios "
    "de OpenShift u otros productos. Responde de manera estructurada, limpia y técnica. "
    "Si no encuentras el SKU, indícalo claramente.",
)
infra_agent_instructions = os.getenv(
    "INFRAESTRUCTURE_AGENT_INSTRUCTIONS",
    (
        "Eres el Arquitecto de Infraestructura Especialista en OpenShift (OCP).\n\n"
        "Responde de manera directa y concisa. Identifica el tipo de infraestructura que "
        "menciona el usuario (on-prem, virtualización, nube pública, híbrido, edge) y evalúa "
        "su compatibilidad con Red Hat OpenShift. Incluye requisitos de nodos control plane "
        "y workers cuando la consulta pida dimensionamiento. Una vez entregada la respuesta, "
        "detente; no repitas conclusiones."
    ),
)
finalize_agent_instructions = os.getenv(
    "FINALIZE_AGENT_INSTRUCTIONS",
    (
        "Eres el Director de Orquestación de IA. Los agentes especialistas ya aportaron "
        "información en el historial.\n\n"
        "Organiza sus respuestas en una sola respuesta coherente para el usuario. "
        "DEBES conservar íntegramente todos los datos técnicos: SKUs, precios, tablas, "
        "cantidades, unidades de medida, condiciones y cálculos. No resumas ni omitas "
        "secciones. No repitas información idéntica. Responde en el mismo idioma que el usuario."
    ),
)
temperature = _float_env("TEMPERATURE", "0.95")
supervisor_temperature = _float_env("SUPERVISOR_TEMPERATURE", "0.1")
top_p = _float_env("TOP_P", "0.95")
max_completion_tokens = _int_env("MAX_COMPLETION_TOKENS", "4096")
max_routing_iterations = _int_env("MAX_ROUTING_ITERATIONS", "10")
graph_recursion_limit = _int_env("GRAPH_RECURSION_LIMIT", "15")

print(f"Base URL: {base_url}")
print(f"Model ID: {model_id}")
print(f"Embedding Model: {embedding_model}")
print(f"Temperature: {temperature}")
print(f"Supervisor Temperature: {supervisor_temperature}")
print(f"Top P: {top_p}")
print(f"Max Completion Tokens: {max_completion_tokens}")
print(f"Graph Recursion Limit: {graph_recursion_limit}")

# 1. CONFIGURACIÓN DE LLM Y EMBEDDINGS
client = LlamaStackClient(base_url=base_url)

llm = ChatOpenAI(
    base_url=base_url + "/v1",
    api_key=api_key,
    model_name=model_id,
    temperature=temperature,
    top_p=top_p,
    max_completion_tokens=max_completion_tokens,
)

llm_supervisor = ChatOpenAI(
    base_url=base_url + "/v1",
    api_key=api_key,
    model_name=model_id,
    temperature=supervisor_temperature,
    top_p=top_p,
    max_completion_tokens=max_completion_tokens,
)

embeddings = OpenAIEmbeddings(
    model=embedding_model,
    openai_api_base=base_url + "/v1",
    openai_api_key=api_key,
)

# ==========================================
# 2. DEFINICIÓN DE HERRAMIENTAS (RAG MILVUS)
# ==========================================


def _milvus_uri() -> str:
    explicit = os.getenv("MILVUS_URI", "").strip()
    if explicit:
        return explicit
    host = os.getenv("MILVUS_HOST", "127.0.0.1").strip()
    port = os.getenv("MILVUS_PORT", "19530").strip()
    if host.startswith("http://") or host.startswith("https://"):
        return f"{host}:{port}"
    return f"http://{host}:{port}"


@tool
def herramienta_rag_skus(query: str) -> str:
    """Consulta la base de datos Milvus para buscar códigos de producto (SKUs), disponibilidad, inventario y precios de Red Hat."""

    vector_db_name_ocp = os.getenv("VECTOR_DB_NAME_OCP", "skus_rh_vector_db")
    milvus_uri = _milvus_uri()

    print(f"--- Query: '{query}'")

    texto_busqueda = ""
    if isinstance(query, dict):
        texto_busqueda = (
            query.get("query") or query.get("text") or query.get("arguments") or str(query)
        )
    else:
        texto_busqueda = str(query).strip()

    print(f"--- Generando Embeddings --- Texto limpio para vectorizar: '{texto_busqueda}'")

    try:
        milvus_client = MilvusClient(uri=milvus_uri)

        query_vector = client.embeddings.create(
            input=texto_busqueda, model=embedding_model
        ).data[0].embedding

        results = milvus_client.search(
            collection_name=vector_db_name_ocp,
            data=[query_vector],
            limit=3,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=["text"],
        )

        if not results or len(results[0]) == 0:
            return "No se encontró coincidencia de SKUs en la base de datos vectorial de Milvus."

        chunks = []
        for hit in results[0]:
            texto = hit["entity"].get("text", "")
            chunks.append(f"[Doc SKU]: {texto}")

        contexto = "\n\n".join(chunks)
        return f"[Resultados de Inventario Milvus]:\n{contexto}"
    except Exception as e:
        return f"Error al conectar con Milvus: {str(e)}"


# ==========================================
# 3. DEFINICIÓN DEL ESTADO Y SUPERVISOR
# ==========================================


def _merge_unique_agents(existing: list[str] | None, new: list[str] | None) -> list[str]:
    return list(dict.fromkeys((existing or []) + (new or [])))


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    next_agent: str
    last_agent: str
    agents_completed: Annotated[list[str], _merge_unique_agents]
    routing_iterations: int


members = ["agente_skus", "agente_infraestructura"]
options = ["FIN"] + members

_INFRA_QUERY_KEYWORDS = (
    "infraestructura",
    "on-prem",
    "on prem",
    "onprem",
    "bare-metal",
    "bare metal",
    "vmware",
    "vsphere",
    "rhv",
    "kvm",
    "proxmox",
    "aws",
    "azure",
    "gcp",
    "nube",
    "privada",
    "pública",
    "publica",
    "híbrido",
    "hibrido",
    "despliegue",
    "compatibilidad",
    "plataforma",
    "topología",
    "topologia",
)

_SKU_QUERY_KEYWORDS = (
    "sku",
    "skus",
    "mw0",
    "catálogo",
    "catalogo",
    "precio",
    "oferta comercial",
    "propuesta comercial",
    "licencia",
    "licenciamiento",
    "core band",
    "cotización",
    "cotizacion",
)

_AGENT_ROUTE_ORDER = ["agente_infraestructura", "agente_skus"]

supervisor_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", coordinator_agent_instructions),
        MessagesPlaceholder(variable_name="messages"),
        ("system", coordinator_agent_delegator_prompt.format(options=options)),
    ]
)


class Router(TypedDict):
    next_agent: Literal["FIN", "agente_skus", "agente_infraestructura"]


supervisor_chain = supervisor_prompt | llm_supervisor.with_structured_output(Router)


def _last_human_query(messages: Sequence[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _query_needs_infra(query: str) -> bool:
    text = query.lower()
    return any(keyword in text for keyword in _INFRA_QUERY_KEYWORDS)


def _query_needs_skus(query: str) -> bool:
    text = query.lower()
    return any(keyword in text for keyword in _SKU_QUERY_KEYWORDS)


def _required_agents_for_query(query: str) -> list[str]:
    needs_infra = _query_needs_infra(query)
    needs_skus = _query_needs_skus(query)

    if needs_infra and needs_skus:
        return list(_AGENT_ROUTE_ORDER)
    if needs_skus:
        return ["agente_skus"]
    if needs_infra:
        return ["agente_infraestructura"]
    return []


def _supervisor_context_messages(state: AgentState) -> list[BaseMessage]:
    msgs = list(state["messages"])
    hints: list[BaseMessage] = []

    completed = state.get("agents_completed") or []
    query = _last_human_query(msgs)
    required = _required_agents_for_query(query)
    pending = [agent for agent in required if agent not in completed]

    if required:
        hints.append(
            SystemMessage(
                content=(
                    f"Dominios detectados en la consulta: {', '.join(required)}. "
                    f"Agentes pendientes en este turno: "
                    f"{', '.join(pending) if pending else 'ninguno'}."
                )
            )
        )

    if completed:
        hints.append(
            SystemMessage(
                content=(
                    f"Estado del flujo: agentes que ya completaron su intervención en este turno: "
                    f"{', '.join(completed)}. "
                    f"{'Aún hay agentes pendientes; no elijas FIN.' if pending else 'Puedes elegir FIN.'}"
                )
            )
        )

    last = state.get("last_agent") or ""
    if last:
        hints.append(
            SystemMessage(
                content=(
                    f"Último agente ejecutado: '{last}'. No lo reasignes; elige FIN u otro agente "
                    "que aún no haya participado."
                )
            )
        )

    return msgs + hints


def supervisor_node(state: AgentState):
    routing_iterations = state.get("routing_iterations", 0) + 1

    if routing_iterations > max_routing_iterations:
        print("--- Supervisor: límite de iteraciones alcanzado, forzando FIN")
        return {"next_agent": "FIN", "routing_iterations": routing_iterations}

    agents_completed = list(state.get("agents_completed") or [])
    last_agent = state.get("last_agent") or ""

    decision = supervisor_chain.invoke({"messages": _supervisor_context_messages(state)})[
        "next_agent"
    ]
    print(f"--- Supervisor decisión LLM: {decision}")

    query = _last_human_query(state["messages"])
    required = _required_agents_for_query(query)
    pending = [agent for agent in required if agent not in agents_completed]

    if pending:
        if decision not in pending or decision in agents_completed:
            decision = pending[0]
            print(
                f"--- Guardrail dominio: consulta requiere {required}, "
                f"derivando a {decision}"
            )
    elif required:
        decision = "FIN"
        print("--- Guardrail dominio: todos los agentes requeridos ya participaron")
    elif decision in agents_completed:
        print(f"--- Guardrail: {decision} ya completó su tarea")
        decision = "FIN"
    elif decision == last_agent and last_agent:
        print(f"--- Guardrail: evitando re-delegar a {last_agent}")
        decision = "FIN"
    elif state["messages"] and isinstance(state["messages"][-1], AIMessage):
        last_name = getattr(state["messages"][-1], "name", None)
        if last_name in members and decision == last_name:
            print(f"--- Guardrail: último mensaje ya es de {last_name}")
            decision = "FIN"

    return {"next_agent": decision, "routing_iterations": routing_iterations}


# ==========================================
# 4. CONFIGURACIÓN DE LOS NODOS ESPECIALISTAS
# ==========================================


def _current_turn_messages(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    last_human_idx = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = i

    turn_messages: list[BaseMessage] = []
    for msg in messages[last_human_idx:]:
        if isinstance(msg, HumanMessage):
            turn_messages.append(msg)
        elif isinstance(msg, AIMessage) and getattr(msg, "name", None) in members:
            turn_messages.append(msg)
    return turn_messages


def skus_agent_node(state: AgentState):
    messages = list(state["messages"])
    print(f"--- Agente SKUs, mensajes: {len(messages)}")
    prompt_skus = SystemMessage(content=skus_agent_instructions)

    query = _last_human_query(messages)
    print(f"--- Agente SKUs: consultando Milvus (query='{query[:120]}')")
    rag_content = herramienta_rag_skus.invoke({"query": query})
    enriched = messages + [
        HumanMessage(
            content=f"[Contexto Milvus — base de conocimiento]\n{rag_content}"
        )
    ]
    response = llm.invoke([prompt_skus] + enriched)
    return {
        "messages": [AIMessage(content=response.content, name="agente_skus")],
        "last_agent": "agente_skus",
        "agents_completed": ["agente_skus"],
    }


def infra_agent_node(state: AgentState):
    messages = state["messages"]
    print(f"--- Agente Infraestructura, mensajes: {len(messages)}")
    prompt_infra = SystemMessage(content=infra_agent_instructions)
    response = llm.invoke([prompt_infra] + list(messages))
    return {
        "messages": [
            AIMessage(content=response.content, name="agente_infraestructura"),
        ],
        "last_agent": "agente_infraestructura",
        "agents_completed": ["agente_infraestructura"],
    }


def finalize_node(state: AgentState):
    turn_messages = _current_turn_messages(state["messages"])
    print(
        f"--- Nodo finalize: sintetizando respuesta final "
        f"({len(turn_messages)} mensajes del turno actual)"
    )
    prompt = SystemMessage(content=finalize_agent_instructions)
    response = llm.invoke([prompt] + turn_messages)
    return {
        "messages": [AIMessage(content=response.content, name="coordinador")],
    }


# ==========================================
# 5. CONSTRUCCIÓN DEL GRAFO DE AGENTES
# ==========================================


def route_supervisor(state: AgentState) -> str:
    next_agent = state.get("next_agent")
    if next_agent != "FIN":
        return next_agent

    completed = state.get("agents_completed") or []
    if len(completed) <= 1:
        print(
            f"--- Supervisor: {len(completed)} agente(s) participó; "
            "omitiendo finalize, usando respuesta del especialista"
        )
        return END

    return "finalize"


def construir_grafo():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("agente_skus", skus_agent_node)
    workflow.add_node("agente_infraestructura", infra_agent_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge("agente_skus", "supervisor")
    workflow.add_edge("agente_infraestructura", "supervisor")

    workflow.add_conditional_edges("supervisor", route_supervisor)
    workflow.add_edge("finalize", END)
    workflow.add_edge(START, "supervisor")

    return workflow.compile()


def _initial_graph_state(chat_history: list[BaseMessage]) -> AgentState:
    return {
        "messages": list(chat_history),
        "next_agent": "",
        "last_agent": "",
        "agents_completed": [],
        "routing_iterations": 0,
    }


# ==========================================
# 6. INTERFAZ EN CHAINLIT
# ==========================================


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("graph", construir_grafo())
    cl.user_session.set("chat_history", [])


@cl.set_starters
async def set_starters():
    starters = [
        cl.Starter(
            label="Consultar SKU de ACM",
            message=(
                "Puedes listar toda la información de SKUs que tengas para el producto "
                "Advanced Cluster Management?"
            ),
            icon="/public/circle-question-mark.svg",
        ),
        cl.Starter(
            label="Tipos de infraestructura para Openshift",
            message=(
                "Puedes listar todos los tipos de infraestructura sobre los que puede correr Red Hat Openshift?"
            ),
            icon="/public/circle-question-mark.svg",
        ),
    ]
    return starters


@cl.on_message
async def on_message(message: cl.Message):
    graph = cl.user_session.get("graph")
    chat_history = cl.user_session.get("chat_history")

    chat_history.append(HumanMessage(content=message.content))

    root_msg = None
    texto_respuesta_usuario = ""
    ultima_respuesta_especialista = ""
    graph_config = {"recursion_limit": graph_recursion_limit}

    async for chunk in graph.astream(
        _initial_graph_state(chat_history),
        config=graph_config,
    ):
        for node, data in chunk.items():
            print(f"Node: {node}")
            print(f"Data: {data}")

            if node == "supervisor":
                next_agent = data.get("next_agent", "FIN")
                if next_agent != "FIN":
                    async with cl.Step(
                        name="Supervisor pensando...",
                        type="tool",
                        icon="bot-message-square",
                    ) as step:
                        step.output = (
                            f"Analizando la intención del usuario... "
                            f"Derivando el flujo hacia: **{next_agent}**."
                        )
                else:
                    async with cl.Step(
                        name="Supervisor respondiendo...",
                        type="tool",
                        icon="bot-message-square",
                    ) as step:
                        step.output = (
                            "Flujo completado. Sintetizando la respuesta final para el usuario."
                        )

            elif node == "agente_skus":
                if "messages" in data:
                    last_msg = data["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        ultima_respuesta_especialista = last_msg.content
                    async with cl.Step(
                        name="Agente de SKUs respondiendo...",
                        type="tool",
                        icon="database-search",
                    ) as step:
                        step.output = (
                            "Consulté Milvus y preparé la respuesta del catálogo de SKUs."
                        )

            elif node == "agente_infraestructura":
                if "messages" in data:
                    last_msg = data["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        ultima_respuesta_especialista = last_msg.content
                    async with cl.Step(
                        name="Agente de Infraestructura respondiendo...",
                        type="tool",
                        icon="bot-message-square",
                    ) as step:
                        step.output = (
                            "Respuesta del Agente de Infraestructura lista para síntesis."
                        )

            elif node == "finalize":
                if "messages" in data:
                    last_msg = data["messages"][-1]
                    if isinstance(last_msg, AIMessage):
                        texto_respuesta_usuario = last_msg.content
                        async with cl.Step(
                            name="Coordinador sintetizando respuesta...",
                            type="tool",
                            icon="bot-message-square",
                        ) as step:
                            step.output = "Respuesta final estructurada para el usuario."

            if "messages" in data:
                chat_history.extend(data["messages"])

    cl.user_session.set("chat_history", chat_history)
    texto_final = texto_respuesta_usuario or ultima_respuesta_especialista
    if texto_final and not root_msg:
        root_msg = cl.Message(content="")
        for token in texto_final.replace("\n", " \n ").split(" "):
            await root_msg.stream_token(token + " ")
        await root_msg.send()
