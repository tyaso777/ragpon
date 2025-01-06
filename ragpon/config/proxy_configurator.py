# %%
import getpass
import os
from urllib.parse import quote

import truststore

from ragpon._utils.logging_helper import get_library_logger

# Initialize logger
logger = get_library_logger(__name__)


class ProxyConfigurator:
    def __init__(self):
        self.proxies = None

    def use_truststore(self):
        """
        Configure proxies to use system's truststore credentials.
        """
        try:
            truststore.inject_into_ssl()
            self.proxies = None
            logger.info("Proxies cleared; truststore will be used.")
        except Exception as e:
            logger.error(f"Failed to use truststore: {e}")
            raise RuntimeError("Truststore configuration failed.") from e

    def use_manual_credentials(self, username: str, password: str, proxy_url: str):
        """
        Manually configure proxies with provided credentials.
        """
        self.proxies = {
            "http": f"http://{quote(username)}:{quote(password)}@{proxy_url}",
            "https": f"https://{quote(username)}:{quote(password)}@{proxy_url}",
        }
        logger.info("Manual credentials set for proxy configuration.")

    def use_getpass_credentials(self, proxy_url: str):
        """
        Configure proxies by prompting the user to input their credentials.
        """
        try:
            username = input("Enter your proxy username (required): ")
            if not username:
                raise ValueError("Username cannot be empty.")
            password = getpass.getpass("Enter your proxy password (required): ")
            if not password:
                raise ValueError("Password cannot be empty.")
            self.proxies = {
                "http": f"http://{quote(username)}:{quote(password)}@{proxy_url}",
                "https": f"https://{quote(username)}:{quote(password)}@{proxy_url}",
            }
            logger.info("Credentials provided via getpass for proxy configuration.")
        except ValueError as ve:
            logger.error(f"Invalid input: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error collecting credentials: {e}")
            raise RuntimeError("Failed to configure proxies with getpass.") from e

    def use_environment_variables(self):
        """
        Configure proxies using environment variables.
        """
        try:
            proxy_url = os.getenv("HTTP_PROXY")
            proxy_url_secure = os.getenv("HTTPS_PROXY")
            if not proxy_url or not proxy_url_secure:
                raise ValueError(
                    "HTTP_PROXY or HTTPS_PROXY environment variables are not set."
                )
            self.proxies = {
                "http": proxy_url,
                "https": proxy_url_secure,
            }
            logger.info("Proxies configured using environment variables.")
        except Exception as e:
            logger.error(f"Failed to configure proxies from environment variables: {e}")
            raise RuntimeError(
                "Failed to configure proxies from environment variables."
            ) from e

    def get_proxies(self):
        """
        Return the configured proxies.
        """
        if self.proxies is None:
            logger.warning(
                "No proxies explicitly configured. Truststore or system defaults will be used."
            )
        return self.proxies
