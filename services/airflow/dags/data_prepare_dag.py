
from airflow.models import Variable
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta

from airflow.utils.dates import days_ago
import sys
import os


# BASE_PATH = os.path.expandvars("$PROJECTPATH")
BASE_PATH = 'data/games.csv'
PROJECT_ROOT = Variable.get("PROJECT_ROOT")


# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'data_prepare_dag',
    default_args=default_args,
    description='A simple data extraction and validation DAG',
    schedule_interval='*/30 * * * *',
    catchup=False,
)

t1 = ExternalTaskSensor(
        task_id='wait_for_data_extract',
        external_dag_id='data_extract_dag', 
        external_task_id='load_data', # wait for the last task in the data_extract_dag DAG
        # execution_delta = timedelta(minutes=10),
        dag=dag
)

t2 = BashOperator(
        task_id='run_zml_pipeline',
        bash_command=f'python {PROJECT_ROOT}/data/extract_data.py',
        dag=dag
)

t1 >> t2
