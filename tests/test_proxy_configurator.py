import logging
import os
from unittest.mock import MagicMock, patch

import pytest
import truststore
from pytest import LogCaptureFixture

from ragpon.config.proxy_configurator import ProxyConfigurator


@pytest.fixture
def configurator() -> ProxyConfigurator:
    """
    Fixture to provide a fresh instance of ProxyConfigurator for each test.
    """
    return ProxyConfigurator()


def test_truststore_injection_success(
    mocker: MagicMock, caplog: LogCaptureFixture, configurator: ProxyConfigurator
) -> None:
    """
    Test that use_truststore calls truststore.inject_into_ssl and clears proxies.
    """
    with caplog.at_level(logging.INFO):
        mock_inject = mocker.patch.object(
            truststore, "inject_into_ssl", return_value=None
        )
        configurator.use_truststore()

        assert configurator.get_proxies() is None
        mock_inject.assert_called_once()
        assert "Proxies cleared; truststore will be used." in caplog.text


def test_truststore_injection_error(
    mocker: MagicMock, caplog: LogCaptureFixture, configurator: ProxyConfigurator
) -> None:
    """
    Test that use_truststore raises RuntimeError if truststore.inject_into_ssl fails.
    """
    mock_inject = mocker.patch.object(
        truststore, "inject_into_ssl", side_effect=Exception("Test error")
    )

    with pytest.raises(RuntimeError, match="Truststore configuration failed."):
        configurator.use_truststore()
    mock_inject.assert_called_once()
    assert "Failed to use truststore: Test error" in caplog.text


def test_manual_credentials_success(configurator: ProxyConfigurator) -> None:
    """
    Test that manual credentials are URL-encoded and stored in the proxies dict.
    """
    configurator.use_manual_credentials(
        "user name", "p@ss word", "proxy.example.com:8080"
    )
    proxies = configurator.get_proxies()

    assert proxies["http"] == "http://user%20name:p%40ss%20word@proxy.example.com:8080"
    assert (
        proxies["https"] == "https://user%20name:p%40ss%20word@proxy.example.com:8080"
    )


@patch("getpass.getpass", side_effect=["testpass"])
@patch("builtins.input", side_effect=["testuser"])
def test_getpass_credentials_success(
    mock_input: MagicMock, mock_getpass: MagicMock, configurator: ProxyConfigurator
) -> None:
    """
    Test that use_getpass_credentials prompts the user and configures proxies.
    """
    configurator.use_getpass_credentials("proxy.example.com:8080")
    proxies = configurator.get_proxies()

    assert proxies["http"] == "http://testuser:testpass@proxy.example.com:8080"
    assert proxies["https"] == "https://testuser:testpass@proxy.example.com:8080"
    mock_input.assert_called_with("Enter your proxy username (required): ")
    mock_getpass.assert_called_once()


@patch("builtins.input", side_effect=[""])
def test_getpass_credentials_empty_username(
    mock_input: MagicMock, configurator: ProxyConfigurator
) -> None:
    """
    Test that use_getpass_credentials raises ValueError if username is empty.
    """
    with pytest.raises(ValueError, match="Username cannot be empty."):
        configurator.use_getpass_credentials("proxy.example.com:8080")


@patch("getpass.getpass", return_value="")
@patch("builtins.input", return_value="testuser")
def test_getpass_credentials_empty_password(
    mock_input: MagicMock, mock_getpass: MagicMock, configurator: ProxyConfigurator
) -> None:
    """
    Test that use_getpass_credentials raises ValueError if password is empty.
    """
    with pytest.raises(ValueError, match="Password cannot be empty."):
        configurator.use_getpass_credentials("proxy.example.com:8080")


@patch.dict(
    os.environ,
    {
        "HTTP_PROXY": "http://env.proxy.example.com:8080",
        "HTTPS_PROXY": "https://env.proxy.example.com:8081",
    },
    clear=True,
)
def test_environment_variables_success(configurator: ProxyConfigurator) -> None:
    """
    Test that environment variables are used if HTTP_PROXY and HTTPS_PROXY are set.
    """
    configurator.use_environment_variables()
    proxies = configurator.get_proxies()

    assert proxies["http"] == "http://env.proxy.example.com:8080"
    assert proxies["https"] == "https://env.proxy.example.com:8081"


@patch.dict(os.environ, {}, clear=True)
def test_environment_variables_missing(
    caplog: LogCaptureFixture, configurator: ProxyConfigurator
) -> None:
    """
    Test that use_environment_variables raises RuntimeError if env vars are missing.
    """
    with pytest.raises(
        RuntimeError, match="Failed to configure proxies from environment variables."
    ):
        configurator.use_environment_variables()

    assert "Failed to configure proxies from environment variables" in caplog.text


def test_no_proxy_configuration(
    caplog: LogCaptureFixture, configurator: ProxyConfigurator
) -> None:
    """
    Test that get_proxies logs a warning when no proxy configuration is set.
    """
    proxies = configurator.get_proxies()

    assert proxies is None
    assert (
        "No proxies explicitly configured. Truststore or system defaults will be used."
        in caplog.text
    )
