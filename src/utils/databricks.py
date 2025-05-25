import warnings
from typing import TYPE_CHECKING, Any, Optional
from langchain_community.chat_models import ChatDatabricks
from src.utils.models import (
    DATABRICKS_CLIENT_WARNING,
    InferenceConfig,
    DatabricksModelId,
    ThinkingConfig,
)


if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
else:
    WorkspaceClient = object


def get_databricks_client(
    host: str,
    token: str,
) -> WorkspaceClient:
    """Get a Databricks client.

    Creates a workspace client for interacting with Databricks.

    Args:
        host (str): Databricks workspace URL (e.g., 'https://dbc-123a-xyz.cloud.databricks.com')
        token (str): Databricks personal access token

    Returns:
        WorkspaceClient: Databricks workspace client
    """
    from databricks.sdk import WorkspaceClient
    
    return WorkspaceClient(
        host=host,
        token=token,
    )


def get_chat_model(
    model_id: DatabricksModelId,
    inference_config: InferenceConfig | None = None,
    client: WorkspaceClient | None = None,
    host: str | None = None,
    token: str | None = None,
    endpoint_name: str | None = None,
    thinking_config: Optional[ThinkingConfig] = None,
) -> ChatDatabricks:
    """Get a ChatDatabricks model.

    Args:
        model_id (DatabricksModelId): Model ID
        inference_config (InferenceConfig | None): Inference config
        client (WorkspaceClient | None): Databricks client
        host (str | None): Databricks workspace URL
        token (str | None): Databricks personal access token
        endpoint_name (str | None): Databricks endpoint name (defaults to model ID value if not provided)
        thinking_config (ThinkingConfig | None): Thinking config

    Returns:
        ChatDatabricks: ChatDatabricks model
    """
    if client and (host or token):
        warnings.warn(DATABRICKS_CLIENT_WARNING)

    _client = client
    if not _client and (host and token):
        _client = get_databricks_client(host=host, token=token)
    
    # Use model_id value as the endpoint name if not explicitly provided
    _endpoint_name = endpoint_name or model_id.value

    additional_kwargs = {}
    # Add thinking config if provided
    if thinking_config:
        additional_kwargs['thinking'] = thinking_config.model_dump()

    # Create the ChatDatabricks with appropriate parameters
    if inference_config is None:
        return ChatDatabricks(
            endpoint=_endpoint_name,
            databricks_client=_client,
            model_kwargs=additional_kwargs,
        )

    # If inference_config is provided, include temperature and max_tokens
    return ChatDatabricks(
        endpoint=_endpoint_name,
        databricks_client=_client,
        temperature=inference_config.temperature,
        max_tokens=inference_config.max_tokens,
        model_kwargs=additional_kwargs,
    )
