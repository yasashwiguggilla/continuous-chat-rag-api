import azure.functions as func
import json
import os
from datetime import datetime, timezone

import tiktoken
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from openai import OpenAI


# ============================================================
# FUNCTION APP
# ============================================================

app = func.FunctionApp(
    http_auth_level=func.AuthLevel.ANONYMOUS
)


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

CHAT_API_KEY = os.environ["CHAT_API_KEY"]

CHAT_MODEL = os.environ.get(
    "CHAT_MODEL",
    "openai/gpt-oss-20b"
)

CHAT_SYSTEM_PROMPT = os.environ.get(
    "CHAT_SYSTEM_PROMPT",
    "You are a helpful AI assistant. "
    "Answer clearly and accurately. "
    "If you do not know something, say so."
)

# Model context window
MODEL_CONTEXT_WINDOW = int(
    os.environ.get(
        "CHAT_CONTEXT_WINDOW_TOKENS",
        "131072"
    )
)

# Maximum output tokens
MAX_OUTPUT_TOKENS = int(
    os.environ.get(
        "CHAT_MAX_OUTPUT_TOKENS",
        "2048"
    )
)

# Safety reserve
CONTEXT_RESERVE_TOKENS = int(
    os.environ.get(
        "CHAT_CONTEXT_RESERVE_TOKENS",
        "8192"
    )
)


# ============================================================
# TOKENIZER
# ============================================================

try:
    tokenizer = tiktoken.encoding_for_model(
        CHAT_MODEL
    )
except Exception:
    tokenizer = tiktoken.get_encoding(
        "o200k_harmony"
    )


# ============================================================
# STUDENT STORAGE
# ============================================================

student_connection_string = os.environ[
    "AzureWebJobsStorage"
]

student_blob_service_client = (
    BlobServiceClient.from_connection_string(
        student_connection_string
    )
)

student_container_client = (
    student_blob_service_client.get_container_client(
        "students"
    )
)

student_blob_client = (
    student_container_client.get_blob_client(
        "students.json"
    )
)


def read_students():
    data = (
        student_blob_client
        .download_blob()
        .readall()
    )

    return json.loads(data)


def write_students(students):

    data = json.dumps(
        students,
        indent=2
    )

    student_blob_client.upload_blob(
        data,
        overwrite=True
    )


# ============================================================
# STUDENT APIs
# ============================================================

@app.route(
    route="students",
    methods=["GET"]
)
def get_students(
    req: func.HttpRequest
) -> func.HttpResponse:

    try:

        students = read_students()

        return func.HttpResponse(
            json.dumps(students),
            status_code=200,
            mimetype="application/json"
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": "Unable to read students"
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(
    route="students/{student_id}",
    methods=["GET"]
)
def get_student(
    req: func.HttpRequest
) -> func.HttpResponse:

    try:

        student_id = req.route_params.get(
            "student_id"
        )

        students = read_students()

        for student in students:

            if str(student.get("id")) == str(
                student_id
            ):

                return func.HttpResponse(
                    json.dumps(student),
                    status_code=200,
                    mimetype="application/json"
                )

        return func.HttpResponse(
            json.dumps({
                "error": "Student not found"
            }),
            status_code=404,
            mimetype="application/json"
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": "Unable to retrieve student"
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(
    route="students",
    methods=["POST"]
)
def create_student(
    req: func.HttpRequest
) -> func.HttpResponse:

    try:

        body = req.get_json()

        students = read_students()

        students.append(body)

        write_students(students)

        return func.HttpResponse(
            json.dumps(body),
            status_code=201,
            mimetype="application/json"
        )

    except ValueError:

        return func.HttpResponse(
            json.dumps({
                "error": "Invalid JSON"
            }),
            status_code=400,
            mimetype="application/json"
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": "Unable to create student"
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(
    route="students/{student_id}",
    methods=["PUT"]
)
def update_student(
    req: func.HttpRequest
) -> func.HttpResponse:

    try:

        student_id = req.route_params.get(
            "student_id"
        )

        body = req.get_json()

        students = read_students()

        for index, student in enumerate(
            students
        ):

            if str(student.get("id")) == str(
                student_id
            ):

                students[index] = body

                write_students(students)

                return func.HttpResponse(
                    json.dumps(body),
                    status_code=200,
                    mimetype="application/json"
                )

        return func.HttpResponse(
            json.dumps({
                "error": "Student not found"
            }),
            status_code=404,
            mimetype="application/json"
        )

    except ValueError:

        return func.HttpResponse(
            json.dumps({
                "error": "Invalid JSON"
            }),
            status_code=400,
            mimetype="application/json"
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": "Unable to update student"
            }),
            status_code=500,
            mimetype="application/json"
        )


@app.route(
    route="students/{student_id}",
    methods=["DELETE"]
)
def delete_student(
    req: func.HttpRequest
) -> func.HttpResponse:

    try:

        student_id = req.route_params.get(
            "student_id"
        )

        students = read_students()

        for index, student in enumerate(
            students
        ):

            if str(student.get("id")) == str(
                student_id
            ):

                deleted_student = students.pop(
                    index
                )

                write_students(students)

                return func.HttpResponse(
                    json.dumps({
                        "message": (
                            "Student deleted successfully"
                        ),
                        "student": deleted_student
                    }),
                    status_code=200,
                    mimetype="application/json"
                )

        return func.HttpResponse(
            json.dumps({
                "error": "Student not found"
            }),
            status_code=404,
            mimetype="application/json"
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": "Unable to delete student"
            }),
            status_code=500,
            mimetype="application/json"
        )


# ============================================================
# GROQ CLIENT
# ============================================================

groq_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# CHAT BLOB STORAGE
# ============================================================

chat_connection_string = os.environ[
    "CHAT_STORAGE_CONNECTION_STRING"
]

chat_blob_service_client = (
    BlobServiceClient.from_connection_string(
        chat_connection_string
    )
)

chat_container_client = (
    chat_blob_service_client.get_container_client(
        "chat-history"
    )
)


def read_conversation(session_id):

    blob_client = (
        chat_container_client.get_blob_client(
            f"{session_id}.json"
        )
    )

    try:

        data = (
            blob_client
            .download_blob()
            .readall()
        )

        return json.loads(data)

    except ResourceNotFoundError:

        return {
            "session_id": session_id,
            "messages": []
        }

    except Exception:
        raise


def save_conversation(
    session_id,
    conversation
):

    blob_client = (
        chat_container_client.get_blob_client(
            f"{session_id}.json"
        )
    )

    data = json.dumps(
        conversation,
        indent=2
    )

    blob_client.upload_blob(
        data,
        overwrite=True
    )


# ============================================================
# TOKEN HELPERS
# ============================================================

def count_message_tokens(message):
    """
    Approximate token count for one message.
    """

    content = message.get(
        "content",
        ""
    )

    content_tokens = len(
        tokenizer.encode(
            content,
            disallowed_special=()
        )
    )

    MESSAGE_OVERHEAD = 8

    return (
        content_tokens
        + MESSAGE_OVERHEAD
    )


def count_messages_tokens(messages):

    total_tokens = 0

    for message in messages:

        total_tokens += count_message_tokens(
            message
        )

    return total_tokens


def build_context(
    messages,
    system_prompt,
    history_token_budget
):
    """
    Build the largest recent conversation context
    that fits inside the configured token budget.
    """

    selected_messages = []

    # Newest → oldest
    for message in reversed(messages):

        candidate = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        candidate.extend(
            reversed(selected_messages)
        )

        candidate.append({
            "role": message["role"],
            "content": message["content"]
        })

        token_count = count_messages_tokens(
            candidate
        )

        if token_count > history_token_budget:

            if not selected_messages:

                raise ValueError(
                    "Conversation context is too large"
                )

            break

        selected_messages.append(
            message
        )

    # Oldest → newest
    selected_messages.reverse()

    final_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    final_messages.extend(
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in selected_messages
    )

    return final_messages


# ============================================================
# RAG INDEX API
# ============================================================

@app.route(
    route="rag/index",
    methods=["POST"]
)
def build_rag_index(
    req: func.HttpRequest
) -> func.HttpResponse:

    # Authenticate
    provided_key = req.headers.get(
        "x-api-key"
    )

    if (
        not provided_key
        or provided_key != CHAT_API_KEY
    ):

        return func.HttpResponse(
            json.dumps({
                "error": "Unauthorized"
            }),
            status_code=401,
            mimetype="application/json"
        )

    try:

        # Lazy import.
        # FAISS is loaded only when this endpoint
        # is actually called.
        from rag.retriever import build_index

        result = build_index()

        return func.HttpResponse(
            json.dumps({
                "message": (
                    "RAG index built successfully"
                ),
                **result
            }),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as exc:

        return func.HttpResponse(
            json.dumps({
                "error": "Unable to build RAG index",
                "details": str(exc)
            }),
            status_code=500,
            mimetype="application/json"
        )


# ============================================================
# CHAT API
# ============================================================

@app.route(
    route="chat",
    methods=["POST"]
)
def chat(
    req: func.HttpRequest
) -> func.HttpResponse:

    # ========================================================
    # 1. API KEY AUTHENTICATION
    # ========================================================

    provided_key = req.headers.get(
        "x-api-key"
    )

    if (
        not provided_key
        or provided_key != CHAT_API_KEY
    ):

        return func.HttpResponse(
            json.dumps({
                "error": "Unauthorized"
            }),
            status_code=401,
            mimetype="application/json"
        )

    # ========================================================
    # 2. READ JSON
    # ========================================================

    try:

        body = req.get_json()

    except ValueError:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "Request body must be valid JSON"
                )
            }),
            status_code=400,
            mimetype="application/json"
        )

    # ========================================================
    # 3. READ INPUTS
    # ========================================================

    session_id = body.get(
        "session_id"
    )

    message = body.get(
        "message"
    )

    # ========================================================
    # 4. VALIDATE SESSION ID
    # ========================================================

    if (
        not session_id
        or not isinstance(
            session_id,
            str
        )
    ):

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "session_id is required "
                    "and must be a string"
                )
            }),
            status_code=400,
            mimetype="application/json"
        )

    session_id = session_id.strip()

    if not session_id:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "session_id cannot be empty"
                )
            }),
            status_code=400,
            mimetype="application/json"
        )

    # ========================================================
    # 5. VALIDATE MESSAGE
    # ========================================================

    if (
        not message
        or not isinstance(
            message,
            str
        )
    ):

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "message is required "
                    "and must be a string"
                )
            }),
            status_code=400,
            mimetype="application/json"
        )

    message = message.strip()

    if not message:

        return func.HttpResponse(
            json.dumps({
                "error": "message cannot be empty"
            }),
            status_code=400,
            mimetype="application/json"
        )

    # ========================================================
    # 6. USER MESSAGE LENGTH
    # ========================================================

    if len(message) > 4000:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "message cannot exceed "
                    "4000 characters"
                )
            }),
            status_code=400,
            mimetype="application/json"
        )

    # ========================================================
    # 7. RAG OPTION
    # ========================================================

    use_rag = body.get(
        "use_rag",
        False
    )

    if not isinstance(
        use_rag,
        bool
    ):

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "use_rag must be true or false"
                )
            }),
            status_code=400,
            mimetype="application/json"
        )

    rag_results = []

    # ========================================================
    # 8. RAG RETRIEVAL
    # ========================================================

    if use_rag:

        try:

            # Lazy import.
            # FAISS is only imported when RAG is used.
            from rag.retriever import retrieve

            rag_results = retrieve(
                message,
                top_k=4
            )

        except FileNotFoundError:

            return func.HttpResponse(
                json.dumps({
                    "error": (
                        "RAG index not found. "
                        "Call /api/rag/index first."
                    )
                }),
                status_code=400,
                mimetype="application/json"
            )

        except Exception as exc:

            return func.HttpResponse(
                json.dumps({
                    "error": (
                        "Unable to retrieve "
                        "PDF context"
                    ),
                    "details": str(exc)
                }),
                status_code=500,
                mimetype="application/json"
            )

    # ========================================================
    # 9. READ CONVERSATION
    # ========================================================

    try:

        conversation = read_conversation(
            session_id
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "Unable to read conversation "
                    "history from Blob Storage"
                )
            }),
            status_code=500,
            mimetype="application/json"
        )

    # ========================================================
    # 10. ADD USER MESSAGE
    # ========================================================

    conversation["messages"].append({

        "role": "user",

        "content": message,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()

    })

    # ========================================================
    # 11. CALCULATE HISTORY TOKEN BUDGET
    # ========================================================

    history_token_budget = (
        MODEL_CONTEXT_WINDOW
        - MAX_OUTPUT_TOKENS
        - CONTEXT_RESERVE_TOKENS
    )

    if history_token_budget <= 0:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "Invalid token budget configuration"
                )
            }),
            status_code=500,
            mimetype="application/json"
        )

    # ========================================================
    # 12. BUILD SYSTEM PROMPT
    # ========================================================

    system_prompt = CHAT_SYSTEM_PROMPT

    if use_rag:

        document_context = "\n\n".join(
            [
                (
                    f"Source: {item['source']}\n"
                    f"Chunk ID: {item['chunk_id']}\n"
                    f"{item['text']}"
                )
                for item in rag_results
            ]
        )

        system_prompt = f"""
{CHAT_SYSTEM_PROMPT}

You are answering a question using the uploaded PDF documents.

Use the provided document context when answering.

Rules:
1. Use the document context as the primary source.
2. Do not invent information.
3. If the answer is not present in the documents,
   say:
   "I could not find that information in the uploaded documents."
4. Keep your answer clear and relevant to the user's question.

DOCUMENT CONTEXT:

{document_context}
"""

    # ========================================================
    # 13. BUILD TOKEN-BASED CONTEXT
    # ========================================================

    try:

        recent_messages = build_context(

            conversation["messages"],

            system_prompt,

            history_token_budget

        )

    except ValueError:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "The conversation context "
                    "is too large for the configured "
                    "token budget"
                )
            }),
            status_code=400,
            mimetype="application/json"
        )

    # ========================================================
    # 14. CALL GROQ
    # ========================================================

    try:

        response = (
            groq_client
            .chat
            .completions
            .create(

                model=CHAT_MODEL,

                messages=recent_messages,

                temperature=0.6,

                max_completion_tokens=(
                    MAX_OUTPUT_TOKENS
                ),

                reasoning_effort="medium"

            )
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "Unable to get response "
                    "from AI service"
                )
            }),
            status_code=502,
            mimetype="application/json"
        )

    # ========================================================
    # 15. EXTRACT AI RESPONSE
    # ========================================================

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    # ========================================================
    # 16. ADD AI RESPONSE TO COMPLETE HISTORY
    # ========================================================

    conversation["messages"].append({

        "role": "assistant",

        "content": answer,

        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()

    })

    # ========================================================
    # 17. SAVE COMPLETE HISTORY
    # ========================================================

    try:

        save_conversation(
            session_id,
            conversation
        )

    except Exception:

        return func.HttpResponse(
            json.dumps({
                "error": (
                    "AI response generated, "
                    "but conversation could not "
                    "be saved"
                )
            }),
            status_code=500,
            mimetype="application/json"
        )

    # ========================================================
    # 18. TOKEN USAGE
    # ========================================================

    usage_data = None

    if getattr(
        response,
        "usage",
        None
    ):

        usage_data = {

            "input_tokens": getattr(
                response.usage,
                "prompt_tokens",
                None
            ),

            "output_tokens": getattr(
                response.usage,
                "completion_tokens",
                None
            ),

            "total_tokens": getattr(
                response.usage,
                "total_tokens",
                None
            )

        }

    # ========================================================
    # 19. RESPONSE
    # ========================================================

    return func.HttpResponse(

        json.dumps({

            "session_id": session_id,

            "message": message,

            "response": answer,

            "model": CHAT_MODEL,

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "usage": usage_data,

            "rag": use_rag,

            "sources": [
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"]
                }
                for item in rag_results
            ]

        }),

        status_code=200,

        mimetype="application/json"

    )