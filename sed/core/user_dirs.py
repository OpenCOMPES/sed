from platformdirs import user_config_path
from platformdirs import user_data_path
from platformdirs import user_log_path

USER_CONFIG_PATH = user_config_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
USER_LOG_PATH = user_log_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
USER_DATA_PATH = user_data_path(appname="sed", appauthor="OpenCOMPES", ensure_exists=True)
