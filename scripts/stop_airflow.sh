#!/bin/bash

kill $(ps -ef | grep "airflow" | awk '{print $2}')
