from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import sys
import yaml

# Ensure the path is correct for the project imports
sys.path.append("/home/rafik/Documents/InnoUni/sum24/mlops/mlops-sum24-project")

from src.data import sample_data, validate_initial_data, run_checkpoint

# Define default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'data_extract_dag',
    default_args=default_args,
    description='A simple data extraction and validation DAG',
    schedule_interval='*/30 * * * *',
    catchup=False,
)

def extract_data(**kwargs):
    # from hydra import initialize, compose
    # initialize(config_path="../configs", job_name="sample_data_job")
    cfg = compose(config_name="config")
    sample_df = sample_data()
    kwargs['ti'].xcom_push(key='sample_df', value=sample_df)

def validate_data(**kwargs):
    sample_df = kwargs['ti'].xcom_pull(key='sample_df', task_ids='extract_data')
    validation_result = validate_initial_data(sample_df)
    if not validation_result["success"]:
        raise ValueError("Data validation failed.")

def version_data(**kwargs):
    # sample_df = kwargs['ti'].xcom_pull(key='sample_df', task_ids='extract_data')
    # Save the sample_df to CSV file
    sample_path = '/home/rafik/Documents/InnoUni/sum24/mlops/mlops-sum24-project/data/samples/sample.csv'
    # sample_df.to_csv(sample_path, index=False)
    
    # Add and commit the data version using DVC
    subprocess.run(['dvc', 'add', sample_path], check=True)
    subprocess.run(['dvc', 'commit', sample_path], check=True)
    
    # Update data_version.yaml
    version_file = '/home/rafik/Documents/InnoUni/sum24/mlops/mlops-sum24-project/configs/data_version.yaml'
    with open(version_file, 'r') as f:
        version_data = yaml.safe_load(f)
    
    version_data['version'] = version_data.get('version', 0) + 1
    
    with open(version_file, 'w') as f:
        yaml.safe_dump(version_data, f)

def load_data(**kwargs):
    sample_path = os.path.join(os.path.dirname(__file__), '../data/sample.csv')
    # Push the data to remote storage
    subprocess.run(['dvc', 'push'], check=True)


def validate_and_run_checkpoint():
    validate_initial_data()
    if not run_checkpoint():
        raise Exception("Checkpoint failed!")

# Define the tasks
t1 = PythonOperator(
    task_id='sample_data',
    python_callable=sample_data,
    provide_context=True,
    dag=dag,
)

t2 = PythonOperator(
    task_id='validate_initial_data',
    python_callable=validate_and_run_checkpoint,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='version_data',
    python_callable=version_data,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

# Set the task dependencies
t1 >> t2 >> t3 >> t4

