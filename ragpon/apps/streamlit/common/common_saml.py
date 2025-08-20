"""
common_saml.py  –  SAML bypass for load-testing (URL ?user_id=...)

Add a query-string such as

    http://localhost:8005/?user_id=loaduser-0042&employee_class_id=70

and this module fakes the SAML cookies (`session_token`, `session_data`)
so that the rest of `streamlit_app.py` believes the user is authenticated.
"""

from __future__ import annotations

import json
import urllib.parse
import uuid
from typing import Final

import streamlit as st
from streamlit import components  # built-in HTML component

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
SESSION_TOKEN_COOKIE: Final[str] = "session_token"
SESSION_DATA_COOKIE: Final[str] = "session_data"
DEFAULT_EMPLOYEE_CLASS_ID: Final[str] = "99"


# ----------------------------------------------------------------------
# Helper to emit the <script> that sets cookies in the browser
# ----------------------------------------------------------------------
def _emit_cookie_script(*, user_id: str, employee_class_id: str) -> None:
    """Send JS to client that sets cookies then reloads the page."""
    session_token = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_id))
    attr_json = {
        "employeeNumber": [user_id],
        "employee_class_id": [employee_class_id],
    }
    session_data_val = urllib.parse.quote(json.dumps(attr_json, ensure_ascii=False))

    cookie_js = f"""
    <script>
      // ---- set the two cookies expected by streamlit_app.py ----
      document.cookie = "{SESSION_TOKEN_COOKIE}={session_token}; path=/; SameSite=Lax";
      document.cookie = "{SESSION_DATA_COOKIE}={session_data_val}; path=/; SameSite=Lax";

      // ---- clean the URL (remove user_id/employee_class_id) ----
      const url = new URL(window.parent.location);          // <- parent!
      url.searchParams.delete("user_id");
      url.searchParams.delete("employee_class_id");
      window.parent.history.replaceState(null, "", url);

      // ---- reload the whole Streamlit page, not just the iframe ----
      window.parent.location.reload();                      // <- changed
    </script>
    """

    # Height 0 keeps it invisible; JS runs immediately.
    components.v1.html(cookie_js, height=0)
    # Stop the current script run.  After reload, cookies are present and
    # streamlit_app.py will skip SAML automatically.
    st.stop()


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def login() -> None:
    """Called by streamlit_app.py when SAML cookies are absent.

    If the URL contains ?user_id=..., we create dummy cookies instead of
    redirecting to the real SAML IdP.  Otherwise we fall back to the real
    SAML flow (imported lazily to avoid circular deps).
    """
    qp = st.query_params  # modern, non-experimental API

    if "user_id" not in qp:
        # ---- normal production path: real SAML ----
        from real_saml_handler import login as real_login  # type: ignore

        real_login()
        return

    # ---- load-test path: create fake cookies in the browser ----
    user_id: str = qp["user_id"]
    employee_class_id: str = qp.get("employee_class_id", DEFAULT_EMPLOYEE_CLASS_ID)

    _emit_cookie_script(user_id=user_id, employee_class_id=employee_class_id)
