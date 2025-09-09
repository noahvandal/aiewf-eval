from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

# 1. End Session
end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

# 2. Submit Dietary Request
submit_dietary_request_function = FunctionSchema(
    name="submit_dietary_request",
    description="Submit a dietary request.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person making the request.",
        },
        "dietary_preference": {
            "type": "string",
            "description": "The dietary preference (e.g., vegetarian, gluten-free).",
        },
    },
    required=["name", "dietary_preference"],
)

# 3. Submit Session Suggestion
submit_session_suggestion_function = FunctionSchema(
    name="submit_session_suggestion",
    description="Submit a suggestion for a new session.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person making the suggestion.",
        },
        "suggestion_text": {
            "type": "string",
            "description": "The text of the session suggestion.",
        },
    },
    required=["name", "suggestion_text"],
)

# 4. Vote for a Session
vote_for_session_function = FunctionSchema(
    name="vote_for_session",
    description="Vote for an existing session.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person voting.",
        },
        "session_id": {
            "type": "string",
            "description": "The Session ID of the session being voted for.",
        },
    },
    required=["name", "session_id"],
)

# 5. Request for Tech Support
request_tech_support_function = FunctionSchema(
    name="request_tech_support",
    description="Request technical support.",
    properties={
        "name": {
            "type": "string",
            "description": "The name of the person requesting support.",
        },
        "issue_description": {
            "type": "string",
            "description": "A description of the technical issue.",
        },
    },
    required=["name", "issue_description"],
)

# Create a ToolsSchema with all the functions
ToolsSchemaForTest = ToolsSchema(
    standard_tools=[
        end_session_function,
        submit_dietary_request_function,
        submit_session_suggestion_function,
        vote_for_session_function,
        request_tech_support_function,
    ]
)
