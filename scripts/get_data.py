def get_data(
    project_id=GCP_MONITOR_THE_REACTOR,
    dataset=BQ_FAULTY_TRAIN,
    col_to_keep=0,
    col_to_drop=0,
    sample_division=SAMPLE_DIVISION,
) :

    if col_to_keep == 0:
        query = f"""
        SELECT {', '.join(col_to_drop)}
        FROM `{project_id}`.`{dataset}`.`csv`
        WHERE
        MOD(sample, {sample_division}) = 0
        ORDER BY faultNumber, simulationRun, sample
        """

        client = bigquery.Client(project=project_id, location="EU")
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        return df

    elif col_to_drop == 0:
        query = f"""
        SELECT {', '.join(col_to_keep)}
        FROM `{project_id}`.`{dataset}`.`csv`
        WHERE
        MOD(sample, {sample_division}) = 0
        ORDER BY faultNumber, simulationRun, sample
        """

        client = bigquery.Client(project=project_id, location="EU")
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        return df
