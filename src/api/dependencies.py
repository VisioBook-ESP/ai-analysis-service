from fastapi import Header, HTTPException


async def get_current_user(x_user_id: str = Header(..., alias="x-user-id")) -> str:
    """Extract and validate the x-user-id header set by the Istio gateway.

    The gateway validates the JWT and injects x-user-id from the token's
    ``sub`` claim. VirtualServices strip any client-supplied header to
    prevent spoofing.

    Returns the user ID string (UUID).
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=401, detail="Missing x-user-id header")
    return x_user_id.strip()
