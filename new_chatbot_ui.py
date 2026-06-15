import chainlit as cl
from llama_stack_client import LlamaStackClient, Agent
from llama_stack_client.lib.agents.react.agent import ReActAgent
import os
from dotenv import load_dotenv
import logging
from constants import LOG_LEVELS
import numpy as np

load_dotenv()
root_log_level = os.getenv("ROOT_LOG_LEVEL", "INFO")
app_log_level = os.getenv("APP_LOG_LEVEL", "INFO")

# Set logging levels
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=LOG_LEVELS[root_log_level], format='%(asctime)s - %(levelname)s - %(message)s', force=True)

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVELS[app_log_level])
base_url_llama_stack = os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")
# Inicializamos el cliente fuera de los decoradores para que sea persistente
client = LlamaStackClient(base_url=base_url_llama_stack)

model_id = os.getenv("MODEL_ID", "granite-3-3-8b-instruct")
logger.info(f"Model ID: {model_id}")
logger.info(f"Base URL: {base_url_llama_stack}")

# Function to create the agent with the specified model and tools
def create_agents():
    """Create an agent with the specified model and tools."""

    vector_db_name = os.getenv("VECTOR_DB_NAME_OCP", "ocp_rh_vector_db")
    vector_db_id = ""

    vector_dbs = client.vector_stores.list()
    for vector_db in vector_dbs:
        if vector_db.name == vector_db_name:
            vector_db_id = vector_db.id
            break

    if vector_db_id == "":
        logger.error(f"Vector DB ID for OCP: {vector_db_name} not found in the vector stores")

    logger.info(f"Vector DB ID for OCP: {vector_db_id}")

    vector_db_name = os.getenv("VECTOR_DB_NAME_SKUS", "skus_rh_vector_db")
    vector_db_id_skus = ""
    vector_dbs = client.vector_stores.list()
    for vector_db in vector_dbs:
        if vector_db.name == vector_db_name:
            vector_db_id_skus = vector_db.id
            break
    if vector_db_id == "":
        logger.error(f"Vector DB ID for SKUs: {vector_db_name} not found in the vector stores")

    logger.info(f"Vector DB ID for SKUs: {vector_db_id_skus}")

    proposal_toolgroup_id = os.getenv("OCP_TOOLGROUP_ID", "ocp::proposal")
    agent_instructions = os.getenv("AGENT_INSTRUCTIONS", "You are a helpful assistant.")
    mcp_server_url = os.getenv("MCP_SERVER_OCP_URL", "http://localhost:7860/gradio_api/mcp/sse")
    agent_instructions_coordinador = os.getenv("AGENT_INSTRUCTIONS_COORDINADOR", "You are a helpful assistant that coordinates the work of the other agents.")
    agent_instructions_general = os.getenv("AGENT_INSTRUCTIONS_GENERAL", "You are a helpful assistant.")
    agent_instructions_ocp_infra = os.getenv("AGENT_INSTRUCTIONS_OCP_INFRA", "You are a helpful assistant for infrastructure evaluation and compatibility with Red Hat OpenShift.")
    agent_instructions_ocp_sizing = os.getenv("AGENT_INSTRUCTIONS_OCP_SIZING", "You are a helpful assistant for capacity planning, cluster sizing, and resource optimization.")
    agent_instructions_ocp_skus = os.getenv("AGENT_INSTRUCTIONS_OCP_SKUS", "You are a helpful assistant for SKU.")

    logger.debug(f"MCP Server URL: {mcp_server_url}")
    logger.debug(f"Proposal Toolgroup ID: {proposal_toolgroup_id}")
    logger.debug(f"Agent Instructions: {agent_instructions_coordinador}")

    #initializing the agent
    agents = {
        "coordinador": ReActAgent(
            client=client,
            model=model_id,
            instructions=agent_instructions_coordinador,
        ),
        "openshift":  Agent(
            client=client,
            model=model_id,
            instructions=agent_instructions_general,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_db_id],
                }
            ],
        ),
        "infraestructura":  Agent(
            client=client,
            model=model_id,
            instructions=agent_instructions_ocp_infra,
        ),
        "sizing": Agent(
            client=client,
            model=model_id,
            instructions=agent_instructions_ocp_sizing,
        ),
        "skus": Agent(
            client=client,
            model=model_id,
            instructions=agent_instructions_ocp_skus,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_db_id_skus],
                }
            ],
        )
    }

    sessions = {name: agent.create_session(f"coordination-{name}") for name, agent in agents.items()}

    return agents, sessions

# Function to run the Llama Stack turn
async def run_llama_turn_llm_based_classification(agents, sessions, user_message, root_msg):
    coordinator = agents["coordinador"]
    coordinator_session = sessions["coordinador"]

    turn_stream = coordinator.create_turn(
        session_id=coordinator_session,
        messages=[{"role": "user", "content": user_message}],
        stream=True,
    )

    tool_responses_to_send = []

    for chunk in turn_stream:
        event = chunk.event

        print(f"Event: {event}")
        
        if not hasattr(event, "payload") or event.payload is None:
            continue
            
        payload = event.payload

        # Si el modelo responde directamente con texto
        if payload.event_type == "text_delta":
            await root_msg.stream_token(payload.text)

        # ¡AQUÍ OCURRE LA MAGIA MULTI-CATEGORÍA!
        # Si la query fue mixta, Llama 3.1 enviará este evento MÚLTIPLES veces (una por herramienta)
        elif payload.event_type == "tool_call_requested":
            tool_call = payload.tool_call
            
            async with cl.Step(name=f"Planificador: Activando {tool_call.function_name}") as step:
                step.input = tool_call.arguments
                
                # Ejecutamos la lógica según la herramienta que el LLM decidió activar
                if tool_call.function_name == "skus":
                    print("Skus")
                    resultado = "El módulo OCP cuesta $1,500 USD por core al año."
                elif tool_call.function_name == "infraestructura":
                    print("Infraestructura")
                    resultado = "Para configurar el cluster debes ejecutar: `oc new-project proposal-ai`."
                elif tool_call.function_name == "sizing":
                    print("Sizing")
                    resultado = "Para capacidad de cluster, tamaño de cluster, y optimización de recursos."
                elif tool_call.function_name == "openshift":
                    print("Openshift")
                    resultado = "Para preguntas generales sobre OpenShift."
                else:
                    resultado = "Información no disponible."
                
                step.output = resultado
                
                # Guardamos la respuesta para reanudar el turno del Director
                tool_responses_to_send.append({
                    "call_id": tool_call.call_id,
                    "output": resultado
                })


# Function to run the Llama Stack turn
async def run_llama_turn(agents, sessions, user_message, root_msg):
    """Procesa el flujo de Llama Stack y actualiza la UI de Chainlit en tiempo real."""

    routes = _route_subtask(user_message)

    print(f"Routes: {routes}")

    subtask_results: list[str] = []

    for route in routes:
        agent = agents[route]
        session_id = sessions[route]

        logger.info(f"Running turn for agent: {route} with session: {session_id}")

        turn_stream_internal = agent.create_turn(
            session_id=session_id,
            messages=[{"role": "user", "content": user_message}],
            stream=False,
        )

        text_output = _extract_output(turn_stream_internal)
        subtask_results.append(f"Agent {route} Result: {text_output}")

    synthesis_result = (
        "Collect and detail the following results into a detailed answer to the user's question:\n\n"
        + "\n\n".join(subtask_results)
    )

    logger.info(f"Synthesis Result: {synthesis_result}")

    coordinator = agents["coordinador"]
    coordinator_session = sessions["coordinador"]

    turn_stream = coordinator.create_turn(
        session_id=coordinator_session,
        messages=[{"role": "user", "content": synthesis_result}],
        stream=True,
    )
    
    try:
        for chunk in turn_stream:
            event = chunk.event
            
            #Capturar TEXTO (Deltas de inferencia)
            if hasattr(event, 'delta') and hasattr(event.delta, 'text'):
                await root_msg.stream_token(event.delta.text)

            #Capturar LLAMADAS A HERRAMIENTAS (Como ves en tu log: StepProgress)
            elif event.event_type == 'step_progress':
                print("Step Progress")
                delta = getattr(event, 'delta', None)
                if delta and delta.delta_type == 'tool_call_issued':
                    # Mostramos en Chainlit que se está llamando a una herramienta
                    async with cl.Step(name=f"Ejecutando: {delta.tool_name}") as step:
                        step.input = delta.arguments
                        step.output = "Esperando respuesta del servidor MCP..."

            #Capturar RESULTADOS de herramientas (StepCompleted)
            elif event.event_type == 'step_completed':
                print("Step Completed")
                if hasattr(event, 'result'):
                    # Log interno para ti
                    logger.debug(f"Paso {event.step_id} completado tipo {event.step_type}")

            #Manejo del Payload (Para compatibilidad con otros modelos)
            elif hasattr(event, 'payload'):
                print("Payload")
                payload = event.payload
                if payload.event_type == "text_delta":
                    print("Text Delta")
                    await root_msg.stream_token(payload.text)

    except RuntimeError as e:
        if "No response available" in str(e):
            #Pasa porque el bucle de herramientas falló al final
            await cl.ErrorMessage(content="El agente intentó usar herramientas pero no generó una respuesta final.").send()
        else:
            logger.error(f"RuntimeError: {e}")

def _route_subtask(prompt: str) -> list[str]:
    """
    Route a subtask to the appropriate specialized agent.

    Note: This uses simplistic keyword-based routing for demonstration purposes.
    Production systems should use more sophisticated routing logic, such as:
    - Embedding-based semantic matching
    - LLM-based task classification
    - Multi-turn dialog to clarify ambiguous requests
    """
    lower_prompt = prompt.lower()

    routes_results: list[str] = []

    if any(token in lower_prompt for token in ["openshift"]):
        routes_results.append("openshift")
    if any(token in lower_prompt for token in ["infrastructure"]):
        routes_results.append("infraestructura")
    if any(token in lower_prompt for token in ["sizing"]):
        routes_results.append("sizing")
    if any(token in lower_prompt for token in ["sku"]):
        routes_results.append("skus")
    
    return routes_results

def get_best_agent(user_query):
    knowledge_base = {
        "openshift": ["¿Qué es OpenShift?", "Cómo se instala OpenShift?", "Cómo se configura OpenShift?", "Cómo se administra OpenShift?"],
        "infraestructura": ["¿Qué infraestructura es compatible con OpenShift?", "¿Qué infraestructura es necesaria para OpenShift?"],
        "sizing": ["¿Puedes indicarme un ejemplo de sizing para un cluster de OpenShift?", "¿Qué tamaño de cluster es necesario para OpenShift?"],
        "skus": ["¿Qué skus existen para el producto de OpenShift?", "¿Qué skus existen para", "¿Me puedes sugerir que sku debo recomendar para un cluster de OpenShift?"]
    }

    # 1. Generar embedding de la consulta
    query_embedding = client.embeddings.create(input=user_query, model="sentence-transformers/ibm-granite/granite-embedding-125m-english").data[0].embedding

    best_score = -1
    selected_agent = "coordinador"

    for agent, examples in knowledge_base.items():
        for ex in examples:
            ex_embedding = client.embeddings.create(input=ex, model="sentence-transformers/ibm-granite/granite-embedding-125m-english").data[0].embedding
            
            # Similitud de Coseno simple
            score = np.dot(query_embedding, ex_embedding)
            print(f"Score: {score}")
            print(f"Agent: {agent}")
            if score > best_score:
                best_score = score
                selected_agent = agent
                
    return selected_agent

def _extract_output(response) -> str:
    if isinstance(response, tuple):
        response = response[0]
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text
    output = getattr(response, "output", None)
    return str(output) if output is not None else ""

@cl.on_chat_start
async def on_chat_start():
    """Se ejecuta cuando un usuario abre el chat."""
    #Crear los agentes y la sesión en Llama Stack
    agents, sessions = create_agents()

    #Guardar los IDs de los agentes y la sesión en la sesión del navegador para usarlos luego
    cl.user_session.set("agents", agents)
    cl.user_session.set("sessions", sessions)

    await cl.Message(content="¡Bienvenido! Soy tu asistente de IA para propuestas de Red Hat. ¿En qué puedo ayudarte hoy?").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Se ejecuta cada vez que el usuario envía un mensaje."""
    agents = cl.user_session.get("agents")
    sessions = cl.user_session.get("sessions")

    # Preparamos el mensaje de respuesta vacío para streaming
    root_msg = cl.Message(content="")

    # Ejecutamos la lógica orquestada por el agente director
    await run_llama_turn(agents, sessions, message.content, root_msg)
    #await run_llama_turn_llm_based_classification(agents, sessions, message.content, root_msg)
    
    # Enviamos el mensaje final una vez terminado el streaming
    await root_msg.send()