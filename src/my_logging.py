import datetime
import logging
import os
from pathlib import Path

from azure.monitor.opentelemetry import configure_azure_monitor

APPI_NAMESPACE = "datascience"


def setup_logging(project_afkorting: str):
    """Initiazes the logger."""
    logger = logging.getLogger(project_afkorting)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = Path("logs") / f"run_{timestamp}.txt"
    logpath.parent.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(logpath, mode="w")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    handlers = [file_handler, stream_handler]

    formatter_other = logging.Formatter(
        fmt=f"[%(asctime)s] [{project_afkorting}:%(filename)s:%(lineno)d] %(levelname)s - %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
    )
    for handler in handlers:
        handler.setFormatter(formatter_other)
        logger.addHandler(handler)

    try:
        conn_str = os.environ["APPLICATION_INSIGHTS_CONNECTION_STRING"]
        # this adds an extra handler to logger that logs to APPI:
        enable_appi_logging(project_afkorting, conn_str)
    except Exception as e:
        print(f"\n\n\n Azure handler failed for logger with error: {e}\n\n\n")

    return logger


def enable_appi_logging(name: str, conn_str) -> None:
    """Enable logging handler for Azure Application Insights.

    This function should be called before the first logging message!
    Otherwise it won't have an effect!

    Args:
        name (str): project name that logs will be written under
    """

    opentelemetry_vars = {
        "OTEL_RESOURCE_ATTRIBUTES": f"service.namespace={APPI_NAMESPACE},service.instance.id={name}",
        "OTEL_SERVICE_NAME": name,
        "OTEL_TRACES_SAMPLER_ARG": "0.1",
    }
    for k, v in opentelemetry_vars.items():
        os.environ[k] = v

    configure_azure_monitor(connection_string=conn_str, logger_name=name)


logger = logging.getLogger("vhk")

if __name__ == "__main__":
    logger = setup_logging("verhuiskans")
    logger.info("TEST 1 2 3")
