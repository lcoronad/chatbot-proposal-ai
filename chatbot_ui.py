# chatbot_ui.py
# This file implements a Chatbot UI for interacting with a Red Hat proposal agent using Gradio.
# It allows users to ask questions about Red Hat proposals and receive responses from the agent.
import gradio
import logging
import os
import sys
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient, Agent
from constants import LOG_LEVELS

# Class to create an agentic system for interacting with Llama Stack Agents
#I am a Red Hat associate and I need to know if AWS is the infrastructure recommend to deploy Openshift in the cloud
#What is the size to one cluster in AWS that support 1.000 TPS?
#What SKU I can to sell for 256 vcpus for Red Hat Openshift Container Platform over AWS?
#Can you resume the detail of benefits of AWS, the size of cluster for 1.000 TPS included the recommend instance type in AWS and the SKU for 256 vcpus for Red Hat Openshift Container Platform. Please provide it in the easiest way to understand, preferably with comparative tables
class AgenticProposalRH:
    """Agentic System for interacting with Llama Stack Agents."""

    # Function to initialize the agentic system
    def __init__(self):
        sys.path.append('..')
        # Load environment variables from .env file
        load_dotenv()

        root_log_level = os.getenv("ROOT_LOG_LEVEL", "INFO")
        app_log_level = os.getenv("APP_LOG_LEVEL", "INFO")

        # Set logging levels
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.basicConfig(level=LOG_LEVELS[root_log_level], format='%(asctime)s - %(levelname)s - %(message)s', force=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(LOG_LEVELS[app_log_level])

        self.model_id = os.getenv("MODEL_ID", "granite-3-3-8b-instruct")
        base_url_llama_stack = os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")

        # Initialize the Llama Stack client
        self.client = LlamaStackClient(
            base_url=base_url_llama_stack,
            timeout=120.0,
        )
        self.logger.info(f"Connected to Llama Stack server at {base_url_llama_stack}")

        temperature = float(os.getenv("TEMPERATURE", 0.95))
        if temperature > 0.0:
            top_p = float(os.getenv("TOP_P", 0.95))
            strategy = {"type": "top_p", "temperature": temperature, "top_p": top_p}
        else:
            strategy = {"type": "greedy"}

        max_tokens = int(os.getenv("MAX_TOKENS", 4096))

        # sampling_params will later be used to pass the parameters to Llama Stack Agents/Inference APIs
        self.sampling_params = {
            "strategy": strategy,
            "max_tokens": max_tokens,
        }

        stream_env = os.getenv("STREAM", "True")
        
        # the Boolean 'stream' parameter will later be passed to Llama Stack Agents/Inference APIs
        # any value non equal to 'False' will be considered as 'True'
        self.stream = (stream_env != "False")

        self.max_infer_iters = int(os.getenv("MAX_INFER_ITERS", 1))

        self.logger.info(f"Inference Parameters:\n\tModel: {self.model_id}\n\tSampling Parameters: {self.sampling_params}\n\tstream: {self.stream}\n\tmax_infer_iters: {self.max_infer_iters}")

    # Function to create the agent with the specified model and tools
    def create_agent(self):
        """Create an agent with the specified model and tools."""

        vector_db_id = os.getenv("VECTOR_DB_ID_OCP", "ocp_rh_vector_db")

        self.logger.info(f"Vector DB ID for OCP: {vector_db_id}")

        # Create the agent using the Agent class
        self.agent = Agent(
            client=self.client,
            model=self.model_id,
            instructions=(
                "You are a helpful assistant."
                "You can use the tools available to answer user questions."
            ),
            tools=[
                "ocp::proposal",
                {
                    "name": "builtin::rag/knowledge_search",
                    "args": {"vector_db_ids": [vector_db_id]},
                }
            ],
            input_shields=[],
            output_shields=[],
            max_infer_iters=self.max_infer_iters,
            sampling_params=self.sampling_params,
        )
        self.agent_id = self.agent.agent_id

        # Create a session that will be used to ask the agent a sequence of questions
        self.session_id = self.agent.create_session(session_name="agent1")
        self.history_formatted = []
        self.logger.info(f"Agent created with ID: {self.agent_id} and session ID: {self.session_id}")

    # Function to make questions to the agent and yield responses
    def make_questions(self, question: str, history: list):
        
        self.logger.debug(f"{self.session_id} | Question: {question}")

        # Append the question to the history
        self.history_formatted.append(question)

        # Create a response stream from the agent
        if self.stream:
            response_stream = self.agent.create_turn(
                session_id=self.session_id,
                messages=[
                    {
                        "role": "user", 
                        "content": question
                    }
                ],
                stream=True,
            )

        # Process the response stream and yield partial responses
        partial_response = ""
        for response in response_stream:
            if hasattr(response, "error") and getattr(response, "error", None):

                error_msg = getattr(response, "error", {}).get( "message", "Unknown error" )
                self.logger.debug(f"Error: {error_msg}")
                break
            elif ( hasattr(response, "event") and getattr(response, "event", None)
                and hasattr(response.event, "payload") and response.event.payload.event_type == "turn_complete" ):

                partial_response += response.event.payload.turn.output_message.content
                self.logger.debug(f"{self.session_id} | Response complete")
                yield partial_response

if __name__ == "__main__":
    # Create an instance of the agentic system
    agenticSystem = AgenticProposalRH()
    # Create the agent
    agenticSystem.create_agent()
    # Initialize the gradio chatbot interface
    chatbot_ui = gradio.ChatInterface(
        fn=agenticSystem.make_questions,
        type="messages",
        chatbot=gradio.Chatbot(
            label="Chatbot RH",
            height=600,
            avatar_images=(
                None,
                "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
            ),
            placeholder="<strong>Your corporate generator proposals</strong><br>Ask Me Anything",
            type="messages"),
        title="Generator proposals by Red Hat",
        description="You can ask questions about Red Hat proposals.",
        examples=[
            ["What SKUs for Red Hat Openshift Container Platform you know?"],
            ["What infrastructure are available for Red Hat Openshift Container Platform?"],
            ["Can you detail me if AWS is a valid infrastructure for Red Hat Openshift Container Platform and what SKUs can you provide?"],
            ["I am a Red Hat OpenShift expert. I need to put together a proposal to determine how many servers in an OpenShift cluster I need to install to handle 10,000 transactions per second. What SKU should I use? Please provide it in the easiest way to understand, preferably with comparative tables"],
        ],
        theme=gradio.themes.Soft(primary_hue=gradio.themes.colors.red, secondary_hue=gradio.themes.colors.gray),
        flagging_mode="manual",
        flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    )
    # Launch the chatbot UI
    chatbot_ui.launch(server_name="0.0.0.0", server_port=7861, share=True, debug=True)
