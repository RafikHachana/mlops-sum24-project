from datetime import datetime

from airflow import DAG
from airflow.decorators import task
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.models import Variable
import yaml

# Import custom functions defined in phase 1
from helpers.data_extract_dag_build import extract_sample, validate_sample, version_sample, load_sample

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Define the DAG
dag = DAG(
    'data_extract_pipeline',
    default_args=default_args,
    description='A pipeline to extract, validate, version, and load data',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
)

def extract(**kwargs):
    data = extract_sample()
    # Push the extracted data to XCom
    kwargs['ti'].xcom_push(key='extracted_data', value=data)

def validate(**kwargs):
    # Pull the extracted data from XCom
    data = kwargs['ti'].xcom_pull(key='extracted_data', task_ids='extract_data')
    validated_data = validate_sample(data)
    # Push the validated data to XCom
    kwargs['ti'].xcom_push(key='validated_data', value=validated_data)

def version(**kwargs):
    # Pull the validated data from XCom
    data = kwargs['ti'].xcom_pull(key='validated_data', task_ids='validate_data')
    version_info = version_sample(data)
    # Push the version info to XCom
    kwargs['ti'].xcom_push(key='version_info', value=version_info)

def load(**kwargs):
    # Pull the version info from XCom
    version_info = kwargs['ti'].xcom_pull(key='version_info', task_ids='version_data')
    load_sample(version_info)
    # Update the version number in the config file
    with open('./configs/data_version.yaml', 'w') as file:
        yaml.dump({'version': version_info['version']}, file)

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate,
    provide_context=True,
    dag=dag,
)

version_task = PythonOperator(
    task_id='version_data',
    python_callable=version,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load,
    provide_context=True,
    dag=dag,
)

# Set up dependencies
extract_task >> validate_task >> version_task >> load_task
