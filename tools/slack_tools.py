import os
import requests
from crewai.tools import tool

# Slack API configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_TEAM_ID = os.getenv("SLACK_TEAM_ID")
SLACK_CHANNEL_IDS = os.getenv("SLACK_CHANNEL_IDS")

# Ensure we have the required environment variables
if not SLACK_BOT_TOKEN or not SLACK_TEAM_ID:
    raise ValueError("Please set SLACK_BOT_TOKEN and SLACK_TEAM_ID environment variables")

# Headers for Slack API requests
headers = {
    "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
    "Content-Type": "application/json",
}


@tool("SlackListChannelsTool")
def slack_list_channels(limit: int = 100, cursor: str = None) -> str:
    """
    List public or pre-defined channels in the workspace with pagination.
    Args:
        limit: Maximum number of channels to return (default 100, max 200)
        cursor: Pagination cursor for next page of results
    Returns:
        JSON response from Slack API as a string
    """
    try:
        predefined_channel_ids = SLACK_CHANNEL_IDS
        if not predefined_channel_ids:
            params = {
                "types": ["public_channel"],
                "exclude_archived": True,
                "limit": min(limit, 200),
                "team_id": SLACK_TEAM_ID,
            }
            
            if cursor:
                params["cursor"] = cursor

            response = requests.get(
                "https://slack.com/api/conversations.list",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            return response.text
        else:
            predefined_channel_ids_array = [id.strip() for id in predefined_channel_ids.split(",")]
            channels = []
            
            for channel_id in predefined_channel_ids_array:
                params = {"channel": channel_id}
                response = requests.get(
                    "https://slack.com/api/conversations.info",
                    headers=headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("ok") and data.get("channel") and not data["channel"].get("is_archived"):
                    channels.append(data["channel"])
            
            result = {
                "ok": True,
                "channels": channels,
                "response_metadata": {"next_cursor": ""}
            }
            return str(result)
    except Exception as e:
        return f"Error listing channels: {str(e)}"


@tool("SlackPostMessageTool")
def slack_post_message(channel_id: str, text: str) -> str:
    """
    Post a new message to a Slack channel.
    Args:
        channel_id: The ID of the channel to post to
        text: The message text to post
    Returns:
        JSON response from Slack API as a string
    """
    try:
        payload = {
            "channel": channel_id,
            "text": text,
        }
        
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error posting message: {str(e)}"


@tool("SlackReplyToThreadTool")
def slack_reply_to_thread(channel_id: str, thread_ts: str, text: str) -> str:
    """
    Reply to a specific message thread in Slack.
    Args:
        channel_id: The ID of the channel containing the thread
        thread_ts: The timestamp of the parent message in the format '1234567890.123456'
        text: The reply text
    Returns:
        JSON response from Slack API as a string
    """
    try:
        payload = {
            "channel": channel_id,
            "thread_ts": thread_ts,
            "text": text,
        }
        
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error replying to thread: {str(e)}"


@tool("SlackAddReactionTool")
def slack_add_reaction(channel_id: str, timestamp: str, reaction: str) -> str:
    """
    Add a reaction emoji to a message.
    Args:
        channel_id: The ID of the channel containing the message
        timestamp: The timestamp of the message to react to
        reaction: The name of the emoji reaction (without ::)
    Returns:
        JSON response from Slack API as a string
    """
    try:
        payload = {
            "channel": channel_id,
            "timestamp": timestamp,
            "name": reaction,
        }
        
        response = requests.post(
            "https://slack.com/api/reactions.add",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error adding reaction: {str(e)}"


@tool("SlackGetChannelHistoryTool")
def slack_get_channel_history(channel_id: str, limit: int = 10) -> str:
    """
    Get recent messages from a channel.
    Args:
        channel_id: The ID of the channel
        limit: Number of messages to retrieve (default 10)
    Returns:
        JSON response from Slack API as a string
    """
    try:
        params = {
            "channel": channel_id,
            "limit": limit,
        }
        
        response = requests.get(
            "https://slack.com/api/conversations.history",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting channel history: {str(e)}"


@tool("SlackGetThreadRepliesTool")
def slack_get_thread_replies(channel_id: str, thread_ts: str) -> str:
    """
    Get all replies in a message thread.
    Args:
        channel_id: The ID of the channel containing the thread
        thread_ts: The timestamp of the parent message in the format '1234567890.123456'
    Returns:
        JSON response from Slack API as a string
    """
    try:
        params = {
            "channel": channel_id,
            "ts": thread_ts,
        }
        
        response = requests.get(
            "https://slack.com/api/conversations.replies",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting thread replies: {str(e)}"


@tool("SlackGetUsersTool")
def slack_get_users(limit: int = 100, cursor: str = None) -> str:
    """
    Get a list of all users in the workspace with their basic profile information.
    Args:
        limit: Maximum number of users to return (default 100, max 200)
        cursor: Pagination cursor for next page of results
    Returns:
        JSON response from Slack API as a string
    """
    try:
        params = {
            "limit": min(limit, 200),
            "team_id": SLACK_TEAM_ID,
        }
        
        if cursor:
            params["cursor"] = cursor

        response = requests.get(
            "https://slack.com/api/users.list",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting users: {str(e)}"


@tool("SlackGetUserProfileTool")
def slack_get_user_profile(user_id: str) -> str:
    """
    Get detailed profile information for a specific user.
    Args:
        user_id: The ID of the user
    Returns:
        JSON response from Slack API as a string
    """
    try:
        params = {
            "user": user_id,
            "include_labels": True,
        }
        
        response = requests.get(
            "https://slack.com/api/users.profile.get",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error getting user profile: {str(e)}"