from scripts.params import *
from google.cloud import bigquery

def get_data_test(
    project_id=GCP_PROJECT_NAME,
    dataset=BQ_FAULTY_TRAIN,
    col_to_keep=None,
    col_to_drop=None,
    sample_division=SAMPLE_DIVISION,
    fault=None
) :

    if col_to_keep == None:
        col_left = tuple(set(COLUMN_NAMES) - set(col_to_drop))

        if fault==None:
            query = f"""
            SELECT {', '.join(col_left)}
            FROM `{project_id}`.`{dataset}`.`csv`
                AND MOD(sample, {sample_division}) = 0
            ORDER BY faultNumber, simulationRun, sample
            """

        else:
            query = f"""
            SELECT {', '.join(col_left)}
            FROM `{project_id}`.`{dataset}`.`csv`
            WHERE faultNumber = {fault}
                AND simulationRun = 1
                AND MOD(sample, {sample_division}) = 0
            ORDER BY faultNumber, simulationRun, sample
            """

    elif col_to_drop == None:
        if fault==None:
            query = f"""
            SELECT {', '.join(col_to_keep)}
            FROM `{project_id}`.`{dataset}`.`csv`
                AND MOD(sample, {sample_division}) = 0
            ORDER BY faultNumber, simulationRun, sample
            """

        else:
            query = f"""
            SELECT {', '.join(col_to_keep)}
            FROM `{project_id}`.`{dataset}`.`csv`
            WHERE faultNumber = {fault}
                AND simulationRun = 1
                AND MOD(sample, {sample_division}) = 0
            ORDER BY faultNumber, simulationRun, sample
            """


    client = bigquery.Client(project=project_id, location="EU")
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    return df
