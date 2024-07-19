#!/bin/bash
rm services/airflow/airflow-scheduler.pid
rm services/airflow/airflow-webserver.pid
rm services/airflow/airflow-triggerer.pid
set -a
export AIRFLOW_HOME=$PWD/services/airflow
airflow scheduler --daemon --log-file services/airflow/logs/scheduler.log
airflow webserver --daemon --log-file services/airflow/logs/webserver.log
airflow triggerer --daemon --log-file services/airflow/logs/triggerer.log
