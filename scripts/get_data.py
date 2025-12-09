from scripts.params import *
from google.cloud import bigquery

def get_data(
    project_id=GCP_PROJECT_NAME,
    dataset=BQ_FAULTY_TRAIN,
    col_to_keep=None,
    col_to_drop=None,
    sample_division=SAMPLE_DIVISION,
    number_simulations=NUMBER_SIMULATIONS,
    fault=None
) :

    if col_to_keep == None:
        col_left = tuple(set(COLUMN_NAMES) - set(col_to_drop))

        if fault==None:
            query = f"""
            SELECT {', '.join(col_left)}
            FROM `{project_id}`.`{dataset}`.`csv`
            WHERE MOD(sample, {sample_division}) = 0
                AND simulationRun <= {number_simulations}
            ORDER BY simulationRun, faultNumber, sample
            """

        else:
            query = f"""
            SELECT {', '.join(col_left)}
            FROM `{project_id}`.`{dataset}`.`csv`
            WHERE faultNumber in {fault}
                AND simulationRun = 1
                AND MOD(sample, {sample_division}) = 0
                AND simulationRun <= {number_simulations}
            ORDER BY simulationRun, faultNumber, sample
            """

    elif col_to_drop == None:
        if fault == None:
            query = f"""
            SELECT {', '.join(col_to_keep)}
            FROM `{project_id}`.`{dataset}`.`csv`
            WHERE MOD(sample, {sample_division}) = 0
            AND simulationRun <= {number_simulations}
            ORDER BY simulationRun, faultNumber, sample
            """

        else:
            query = f"""
            SELECT {', '.join(col_to_keep)}
            FROM `{project_id}`.`{dataset}`.`csv`
            WHERE faultNumber in {fault}
                AND simulationRun = 1
                AND MOD(sample, {sample_division}) = 0
                AND simulationRun <= {number_simulations}
            ORDER BY simulationRun, faultNumber, sample
            """


    client = bigquery.Client(project=project_id, location="EU")
    query_job = client.query(query)
    result = query_job.result()
    df = result.to_dataframe()

    return df
